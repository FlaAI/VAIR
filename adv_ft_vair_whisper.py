import os
import sys
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

import robust_speech as rs

logger = get_logger(__name__)

os.environ["PYTHONIOENCODING"] = "utf-8"

import speechbrain.utils.autocast
original_init = speechbrain.utils.autocast.TorchAutocast.__init__

def fixed_init(self, *args, **kwargs):
    if 'device_type' in kwargs and isinstance(kwargs['device_type'], str) and kwargs['device_type'].startswith('cuda:'):
        kwargs['device_type'] = 'cuda'
    return original_init(self, *args, **kwargs)

speechbrain.utils.autocast.TorchAutocast.__init__ = fixed_init


class ASR(sb.Brain):
    def __init__(self, *args, **kwargs):
        self.attacker = kwargs.pop('attacker', None) if 'attacker' in kwargs else None
        self.gaussian_attacker = kwargs.pop('gaussian_attacker', None) if 'gaussian_attacker' in kwargs else None

        super().__init__(*args, **kwargs)

        print(self.modules.whisper)

        self.grad_accumulation_factor = getattr(self.hparams, 'grad_accumulation_factor', 1)
        self.grad_accumulation_step = 0

        self.adv_training_enabled = self.attacker is not None
        if self.adv_training_enabled:
            if hasattr(self.attacker, 'asr_brain'):
                self.attacker.asr_brain = self

        if self.gaussian_attacker is not None:
            if hasattr(self.gaussian_attacker, 'asr_brain'):
                self.gaussian_attacker.asr_brain = self

        self.print_hyps_flag = False
        self.print_adv_hyps_flag = False

        self.large_diff_count = 0
        self.total_samples = 0

        self.feature_similarity_reg_enabled = getattr(self.hparams, 'enable_feature_similarity_reg', False)
        if self.feature_similarity_reg_enabled:
            self.feature_similarity_reg_weight = getattr(self.hparams, 'feature_similarity_reg_weight', 0.1)
            self.feature_similarity_layer_names = getattr(self.hparams, 'feature_similarity_layer_names', [])
            self.feature_similarity_loss_type = getattr(self.hparams, 'feature_similarity_loss_type', 'cosine')
            self.feature_similarity_mode = getattr(self.hparams, 'feature_similarity_mode', 'clean_adv')  # 'clean_adv' or 'gaussian_adv'

            self.layer_hooks = {}
            self.clean_features = {}
            self.gaussian_features = {}
            self.adv_features = {}

            self._register_feature_hooks()

        self.attention_kl_reg_enabled = getattr(self.hparams, 'enable_attention_kl_reg', False)
        if self.attention_kl_reg_enabled:
            self.attention_kl_reg_weight = getattr(self.hparams, 'attention_kl_reg_weight', 0.05)
            self.attention_kl_layer_names = getattr(self.hparams, 'attention_kl_layer_names', [])
            self.attention_kl_temperature = getattr(self.hparams, 'attention_kl_temperature', 1.0)

            self.attention_hooks = {}
            self.clean_attention_weights = {}
            self.adv_attention_weights = {}

            self._debug_attention = getattr(self.hparams, 'debug_attention_extraction', False)

            self._register_attention_hooks()

    def _register_feature_hooks(self):
        def get_hook(layer_name):
            def hook(module, input, output):
                if hasattr(self, '_current_mode'):
                    mode = self._current_mode
                    if mode == 'clean':
                        self.clean_features[layer_name] = output.detach()
                    elif mode == 'gaussian':
                        self.gaussian_features[layer_name] = output.detach()
                    elif mode == 'adv':
                        self.adv_features[layer_name] = output.detach()
                    else:
                        print(f"DEBUG: Unknown mode: {mode}")
            return hook

        for layer_name in self.feature_similarity_layer_names:
            module_parts = layer_name.split('.')
            current_module = self.modules.whisper.model

            for part in module_parts:
                if hasattr(current_module, part):
                    current_module = getattr(current_module, part)
                else:
                    break
            else:
                hook = get_hook(layer_name)
                self.layer_hooks[layer_name] = current_module.register_forward_hook(hook)

    def _register_attention_hooks(self):
        if hasattr(self.modules.whisper, 'model'):
            model_base = self.modules.whisper.model
        else:
            model_base = self.modules.whisper

        try:
            if hasattr(model_base, 'decoder') and hasattr(model_base.decoder, 'layers'):
                for layer_idx, layer in enumerate(model_base.decoder.layers):
                    layer_name = f"decoder.layers.{layer_idx}.encoder_attn"

                    if layer_name in self.attention_kl_layer_names:
                        if hasattr(layer, 'encoder_attn'):
                            if hasattr(layer.encoder_attn, 'attn'):
                                self.attention_hooks[layer_name] = layer.encoder_attn.attn.register_forward_hook(
                                    lambda mod, inp, out, name=layer_name:
                                    self._capture_attention_weights(mod, inp, out, name)
                                )
                            else:
                                self.attention_hooks[layer_name] = layer.encoder_attn.register_forward_hook(
                                    lambda mod, inp, out, name=layer_name:
                                    self._capture_attention_weights(mod, inp, out, name)
                                )

        except Exception as e:
            import traceback
            traceback.print_exc()

    def _capture_attention_weights(self, module, input, output, layer_name):
        try:
            attn_weights = None

            if isinstance(output, tuple) and len(output) >= 2:
                potential_weights = output[1]

                if (isinstance(potential_weights, torch.Tensor) and
                    len(potential_weights.shape) == 4 and  # (batch, heads, seq_q, seq_k)
                    potential_weights.shape[1] > 0):       # heads > 0

                    attn_weights = potential_weights

                    batch_size, num_heads, seq_q, seq_k = attn_weights.shape
                      
            if attn_weights is None:
                if hasattr(module, 'attn_weights') and module.attn_weights is not None:
                    attn_weights = module.attn_weights
                elif hasattr(module, 'attention_probs') and module.attention_probs is not None:
                    attn_weights = module.attention_probs

            if attn_weights is not None:
                if self._current_mode == 'clean':
                    self.clean_attention_weights[layer_name] = attn_weights.detach()
                elif self._current_mode == 'adv':
                    self.adv_attention_weights[layer_name] = attn_weights
            else:
                if hasattr(self, '_debug_attention') and self._debug_attention:
                    if isinstance(output, tuple):
                        for i, item in enumerate(output):
                            if isinstance(item, torch.Tensor):
                                print(f"    [{i}]: Tensor {item.shape}")
                            else:
                                print(f"    [{i}]: {type(item)}")

        except Exception as e:
            if hasattr(self, '_debug_attention') and self._debug_attention:
                import traceback
                traceback.print_exc()

    def _compute_feature_similarity_loss(self, clean_features=None, gaussian_features=None):
        if not self.feature_similarity_reg_enabled:
            return 0.0

        total_loss = 0.0
        loss_count = 0

        for layer_name in self.feature_similarity_layer_names:
            if self.feature_similarity_mode == 'clean_adv':
                if layer_name in clean_features and layer_name in self.adv_features:
                    feat1 = clean_features[layer_name]
                    feat2 = self.adv_features[layer_name]
                    feat1_name = "clean"
                    feat2_name = "adv"
                else:
                    continue
            elif self.feature_similarity_mode == 'gaussian_adv':
                if layer_name in gaussian_features and layer_name in self.adv_features:
                    feat1 = gaussian_features[layer_name]
                    feat2 = self.adv_features[layer_name]
                    feat1_name = "gaussian"
                    feat2_name = "adv"
                else:
                    continue
            else:
                continue

            if feat1.shape != feat2.shape:
                continue

            if self.feature_similarity_loss_type == 'cosine':
                cos_sim = torch.nn.functional.cosine_similarity(
                    feat1.view(feat1.shape[0], -1),
                    feat2.view(feat2.shape[0], -1),
                    dim=1
                )
                loss = 1.0 - cos_sim.mean()
            elif self.feature_similarity_loss_type == 'mse':
                loss = torch.nn.functional.mse_loss(feat1, feat2)
            else:
                loss = torch.nn.functional.mse_loss(feat1, feat2)

            total_loss += loss
            loss_count += 1

        if loss_count > 0:
            return total_loss / loss_count
        else:
            return 0.0

    def _compute_attention_kl_loss(self, clean_attention_weights):
        if not self.attention_kl_reg_enabled:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        total_loss = 0.0
        loss_count = 0

        for layer_name in self.attention_kl_layer_names:
            if layer_name in clean_attention_weights and layer_name in self.adv_attention_weights:
                clean_attn = clean_attention_weights[layer_name]
                adv_attn = self.adv_attention_weights[layer_name]

                if clean_attn is None or adv_attn is None:
                    continue

                if clean_attn.shape != adv_attn.shape:
                    continue

                try:
                    clean_probs = clean_attn + 1e-8
                    adv_probs = adv_attn + 1e-8 
                  
                    kl_loss = torch.nn.functional.kl_div(
                        torch.log(adv_probs),
                        clean_probs,
                        reduction='batchmean'
                    )

                    if torch.isfinite(kl_loss) and not torch.isnan(kl_loss):
                        total_loss += kl_loss
                        loss_count += 1

                except Exception as e:
                    continue

        if loss_count > 0:
            return total_loss / loss_count
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _cleanup_feature_hooks(self):
        for hook in self.layer_hooks.values():
            hook.remove()
        self.layer_hooks.clear()

    def _cleanup_attention_hooks(self):
        for hook in self.attention_hooks.values():
            hook.remove()
        self.attention_hooks.clear()

    def module_train(self):
        """Set PyTorch modules to training mode (required by robust_speech)"""
        self.modules.train()

    def module_eval(self):
        """Set PyTorch modules to eval mode (required by robust_speech)"""
        self.modules.eval()

    def get_tokens(self, predictions):
        """Extract tokens from predictions (required by robust_speech)"""
        return predictions[-1]

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        bos_tokens, bos_tokens_lens = batch.tokens_bos

        abs_tokens_lens = torch.round(
            bos_tokens_lens * bos_tokens.shape[1]
        ).long()
        pad_mask = (
            torch.arange(abs_tokens_lens.max(), device=self.device)[None, :]
            < abs_tokens_lens[:, None]
        )
        bos_tokens[~pad_mask] = self.tokenizer.pad_token_id

        enc_out, logits, _ = self.modules.whisper(wavs, bos_tokens)
        log_probs = self.hparams.log_softmax(logits)

        hyps = None
        if stage == sb.Stage.VALID:
            hyps, _, _, _ = self.hparams.valid_search(
                enc_out.detach(), wav_lens
            )
        elif stage == sb.Stage.TEST:
            hyps, _, _, _ = self.hparams.test_search(enc_out.detach(), wav_lens)

        return log_probs, hyps, wav_lens

    def compute_objectives(self, predictions, batch, stage, adv=False, targeted=False, reduction="mean"):
        """Computes the loss NLL given predictions and targets."""

        (log_probs, hyps, wav_lens) = predictions
        batch = batch.to(self.device)
        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        loss = self.hparams.nll_loss(
            log_probs, tokens_eos, length=tokens_eos_lens
        )

        if stage != sb.Stage.TRAIN and stage != rs.Stage.ATTACK and hyps is not None:
            tokens, tokens_lens = batch.tokens

            predicted_words = [
                self.tokenizer.decode(t, skip_special_tokens=True).strip()
                for t in hyps
            ]

            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer.batch_decode(
                target_words, skip_special_tokens=True
            )

            if (not adv and self.print_hyps_flag) or (adv and self.print_adv_hyps_flag):
                if not adv:
                    self.print_hyps_flag = False
                else:
                    self.print_adv_hyps_flag = False

            if hasattr(self.hparams, "normalized_transcripts"):

                if hasattr(self.tokenizer, "normalize"):
                    normalized_fn = self.tokenizer.normalize
                else:
                    normalized_fn = self.tokenizer._normalize

                predicted_words = [
                    normalized_fn(text).split(" ") for text in predicted_words
                ]

                target_words = [
                    normalized_fn(text).split(" ") for text in target_words
                ]
            else:
                predicted_words = [text.split(" ") for text in predicted_words]
                target_words = [text.split(" ") for text in target_words]

            if adv:
                self.adv_wer_metric.append(ids, predicted_words, target_words)
                self.adv_cer_metric.append(ids, predicted_words, target_words)
            else:
                self.wer_metric.append(ids, predicted_words, target_words)
                self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        if self.adv_training_enabled:
            return self.fit_batch_adversarial(batch)
        else:
            return super().fit_batch(batch)

    def fit_batch_adversarial(self, batch):
        debug_step = (self.step % 1000 == 0)

        if hasattr(self.hparams, "wav_augment"):
            batch.sig = (
                self.hparams.wav_augment(batch.sig[0], batch.sig[1])
            )
            batch.tokens_bos = (
                self.hparams.wav_augment.replicate_labels(batch.tokens_bos[0]),
                self.hparams.wav_augment.replicate_labels(batch.tokens_bos[1])
            )
            batch.tokens_eos = (
                self.hparams.wav_augment.replicate_labels(batch.tokens_eos[0]),
                self.hparams.wav_augment.replicate_labels(batch.tokens_eos[1])
            )

        augmented_wavs = batch.sig[0].requires_grad_(True)
        batch.sig = (augmented_wavs, batch.sig[1])

        if self.feature_similarity_reg_enabled and self.feature_similarity_mode == 'clean_adv':
            self._current_mode = 'clean'
            self.clean_features.clear()
        if self.attention_kl_reg_enabled:
            self._current_mode = 'clean'
            self.clean_attention_weights.clear()
        clean_outputs = self.compute_forward(batch, sb.Stage.TRAIN)
        clean_loss = self.compute_objectives(clean_outputs, batch, sb.Stage.TRAIN, adv=False)

        if self.feature_similarity_reg_enabled and self.feature_similarity_mode == 'clean_adv':
            anchored_features = {
            name: feat.detach() for name, feat in self.clean_features.items()
        }

        if self.attention_kl_reg_enabled:
            anchored_clean_attention_weights = {
                name: feat.detach() for name, feat in self.clean_attention_weights.items()
            }

        if self.grad_accumulation_step == 0:
            self.optimizer.zero_grad()

        clean_loss = clean_loss / self.grad_accumulation_factor
        clean_loss.backward(retain_graph=True)
      
        adv_wavs = self.attacker.perturb(batch)
        self.module_train()

        adv_wavs = adv_wavs.detach().to('cpu').requires_grad_(True)

        if self.feature_similarity_reg_enabled and self.feature_similarity_mode == 'gaussian_adv':
            if self.gaussian_attacker is not None:
                batch.sig = (augmented_wavs, batch.sig[1])

                gaussian_wavs = self.gaussian_attacker.perturb(batch)
                gaussian_wavs = gaussian_wavs.detach().to('cpu').requires_grad_(True)

                if self.feature_similarity_reg_enabled:
                    self._current_mode = 'gaussian'
                    self.gaussian_features.clear()

                batch.sig = (gaussian_wavs, batch.sig[1])
                gaussian_outputs = self.compute_forward(batch, sb.Stage.TRAIN)

                anchored_features = {
                    name: feat.detach() for name, feat in self.gaussian_features.items()
                }

        batch.sig = (adv_wavs, batch.sig[1])
        if self.feature_similarity_reg_enabled:
            self._current_mode = 'adv'
            self.adv_features.clear()
        if self.attention_kl_reg_enabled:
            self._current_mode = 'adv'
            self.adv_attention_weights.clear()
        adv_outputs = self.compute_forward(batch, sb.Stage.TRAIN)
        adv_loss = self.compute_objectives(adv_outputs, batch, sb.Stage.TRAIN, adv=True)

        similarity_loss = 0.0
        if self.feature_similarity_reg_enabled:
            if self.feature_similarity_mode == 'gaussian_adv':
                similarity_loss = self._compute_feature_similarity_loss(gaussian_features=anchored_features)
            elif self.feature_similarity_mode == 'clean_adv':
                similarity_loss = self._compute_feature_similarity_loss(clean_features=anchored_features)

        attention_kl_loss = 0.0
        if self.attention_kl_reg_enabled:
            attention_kl_loss = self._compute_attention_kl_loss(anchored_clean_attention_weights)

        adv_loss = self.hparams.adv_loss_weight * adv_loss
        if self.feature_similarity_reg_enabled:
            adv_loss += self.feature_similarity_reg_weight * similarity_loss
        if self.attention_kl_reg_enabled:
            adv_loss += self.attention_kl_reg_weight * attention_kl_loss

        adv_loss = adv_loss / self.grad_accumulation_factor
      
        adv_loss.backward()

        self.grad_accumulation_step += 1
        if self.grad_accumulation_step >= self.grad_accumulation_factor:
            self.optimizer.step()
            self.grad_accumulation_step = 0
          
        current_batch_loss = clean_loss.detach() * self.grad_accumulation_factor + adv_loss.detach() * self.grad_accumulation_factor
        return current_batch_loss.cpu()

    def evaluate_batch(self, batch, stage):
        if self.adv_training_enabled:
            return self.evaluate_batch_adversarial(batch, stage)
        else:
            return super().evaluate_batch(batch, stage)

    def evaluate_batch_adversarial(self, batch, stage):
        original_wavs = batch.sig[0]

        clean_outputs = self.compute_forward(batch, stage)
        clean_loss = self.compute_objectives(clean_outputs, batch, stage, adv=False)

        with torch.enable_grad():
            original_wavs.requires_grad_(True)
            batch.sig = (original_wavs, batch.sig[1])

            attacker_flag_backup = None
            if hasattr(self.attacker, "train_mode_for_backward"):
                attacker_flag_backup = self.attacker.train_mode_for_backward
                self.attacker.train_mode_for_backward = False

            adv_wavs = self.attacker.perturb(batch)

            if attacker_flag_backup is not None:
                self.attacker.train_mode_for_backward = attacker_flag_backup

        adv_wavs = adv_wavs.detach()

        original_wavs.requires_grad_(False)

        batch.sig = (adv_wavs, batch.sig[1])
        adv_outputs = self.compute_forward(batch, stage)
        adv_loss = self.compute_objectives(adv_outputs, batch, stage, adv=True)

        batch.sig = (original_wavs, batch.sig[1])

        _, clean_hyps, _ = clean_outputs
        _, adv_hyps, _ = adv_outputs

        if clean_hyps is not None and adv_hyps is not None:
            clean_words = [self.tokenizer.decode(t, skip_special_tokens=True).strip().split() for t in clean_hyps]
            adv_words = [self.tokenizer.decode(t, skip_special_tokens=True).strip().split() for t in adv_hyps]

            for clean_len, adv_len in zip(map(len, clean_words), map(len, adv_words)):
                self.total_samples += 1
                if adv_len > 2 * clean_len or adv_len < 0.5 * clean_len:
                    self.large_diff_count += 1

        return (clean_loss + self.hparams.adv_loss_weight * adv_loss).detach().cpu()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage == sb.Stage.TRAIN:
            self.grad_accumulation_step = 0
            param_norm = torch.nn.utils.parameters_to_vector(self.modules.parameters()).norm().item()
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

            if self.adv_training_enabled:
                self.adv_cer_metric = self.hparams.cer_computer()
                self.adv_wer_metric = self.hparams.error_rate_computer()

            self.print_hyps_flag = True
            self.print_adv_hyps_flag = True

            self.large_diff_count = 0
            self.total_samples = 0

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            param_norm = torch.nn.utils.parameters_to_vector(self.modules.parameters()).norm().item()
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

            if self.adv_training_enabled:
                stage_stats["Adv_CER"] = self.adv_cer_metric.summarize("error_rate")
                stage_stats["Adv_WER"] = self.adv_wer_metric.summarize("error_rate")

        if stage == sb.Stage.VALID:
            lr = self.hparams.lr_annealing_whisper.current_lr
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            if self.adv_training_enabled:
                self.checkpointer.save_and_keep_only(
                    meta={"Adv_WER": stage_stats["Adv_WER"]},
                    min_keys=["Adv_WER"],
                )
            else:
                self.checkpointer.save_and_keep_only(
                    meta={"WER": stage_stats["WER"]},
                    min_keys=["WER"],
                )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)
                    if self.adv_training_enabled:
                        w.write("\n\n=== Adversarial WER Results ===\n")
                        self.adv_wer_metric.write_stats(w)


def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_loader_kwargs"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        if (
            "normalized_transcripts" in hparams
            and hparams["normalized_transcripts"]
        ):
            if hasattr(tokenizer, "normalize"):
                normalized_fn = tokenizer.normalize
            else:
                normalized_fn = tokenizer._normalize

            wrd = normalized_fn(wrd)

        yield wrd
        tokens_list = tokenizer.encode(wrd, add_special_tokens=False)
        yield tokens_list
        tokens_list = tokenizer.build_inputs_with_special_tokens(tokens_list)
        tokens_bos = torch.LongTensor(tokens_list[:-1])
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list[1:])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "tokens_list", "tokens_bos", "tokens_eos", "tokens"],
    )

    return train_data, valid_data, test_datasets


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    from data.LibriSpeech.librispeech_prepare import prepare_librispeech  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Defining tokenizer and loading it
    tokenizer = hparams["whisper"].tokenizer

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        opt_class=hparams["whisper_opt_class"],
        attacker=hparams.get("attacker", None),
        gaussian_attacker=hparams.get("gaussian_attacker", None),
    )

    # We load the pretrained whisper model
    if "pretrainer" in hparams.keys():
        hparams["pretrainer"].collect_files()
        hparams["pretrainer"].load_collected(asr_brain.device)

    # We dynamically add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for Whisper.
    asr_brain.tokenizer = tokenizer

    # Ori Testing
    os.makedirs(hparams["output_wer_folder"], exist_ok=True)

    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.test_wer_file = os.path.join(
            hparams["output_wer_folder"], f"wer_{k}.txt"
        )
        asr_brain.evaluate(
            test_datasets[k],
            test_loader_kwargs=hparams["test_loader_kwargs"],
            min_key="WER",
        )

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_loader_kwargs"],
        valid_loader_kwargs=hparams["valid_loader_kwargs"],
    )

    # Testing
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.test_wer_file = os.path.join(
            hparams["output_wer_folder"], f"wer_{k}.txt"
        )
        asr_brain.evaluate(
            test_datasets[k],
            test_loader_kwargs=hparams["test_loader_kwargs"],
            min_key="WER",
        )
