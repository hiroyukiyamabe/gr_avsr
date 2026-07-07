import torch
import torchaudio
from cosine import WarmupCosineScheduler
from datamodule.transforms import get_text_transform

from pytorch_lightning import LightningModule
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_conformer_av import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.scorers.ctc import CTCPrefixScorer


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())


def compute_char_level_distance(seq1, seq2):
    # Character Error Rate unit: edit distance over characters (Japanese / CER).
    return torchaudio.functional.edit_distance(list(seq1), list(seq2))


def _load_frontend(module, path):
    """Load an extracted front-end (``trunk.*`` / ``frontend3D.*``) into ``module``.

    Accepts front-end checkpoints produced by ``module_extractor/extractor.py``
    (wrapped in ``model_state_dict``) as well as raw front-end state dicts.
    """
    ckpt = torch.load(path, map_location=lambda storage, loc: storage)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    tmp_ckpt = {
        k: v for k, v in state.items()
        if k.startswith("trunk.") or k.startswith("frontend3D.")
    }
    module.load_state_dict(tmp_ckpt)


class ModelModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.backbone_args = self.cfg.model.audiovisual_backbone

        self.lang = getattr(self.cfg.data, "lang", "en")
        self.text_transform = get_text_transform(self.lang)
        self.token_list = self.text_transform.token_list
        self.model = E2E(len(self.token_list), self.backbone_args)

        # -- initialise
        # For AVSR cross-lingual transfer (framework (c) in Kondo & Tamura, 2025)
        # we inherit BOTH front-ends and fine-tune them: the visual front-end
        # (3D conv + 2D ResNet) goes into encoder.frontend, and the audio
        # front-end (1D ResNet) into aux_encoder.frontend. They come from
        # separate English checkpoints (VSR and ASR respectively), so we accept
        # two paths and fall back to pretrained_model_path for the visual one.
        if self.cfg.transfer_frontend:
            visual_path = getattr(self.cfg, "pretrained_visual_frontend_path", None) or self.cfg.pretrained_model_path
            audio_path = getattr(self.cfg, "pretrained_audio_frontend_path", None)
            if visual_path:
                _load_frontend(self.model.encoder.frontend, visual_path)
            if audio_path:
                _load_frontend(self.model.aux_encoder.frontend, audio_path)
        elif self.cfg.pretrained_model_path:
            ckpt = torch.load(self.cfg.pretrained_model_path, map_location=lambda storage, loc: storage)
            if self.cfg.transfer_encoder:
                tmp_ckpt = {k.replace("encoder.", ""): v for k, v in ckpt.items() if k.startswith("encoder.")}
                self.model.encoder.load_state_dict(tmp_ckpt, strict=True)
            else:
                # Strict except for the fusion gate: a pre-gate checkpoint loads
                # with fusion_gate.* left at zero-init (neutral 0.5/0.5 gate),
                # which reproduces the original static fusion exactly. Any other
                # mismatch is a genuine error and still raises.
                missing, unexpected = self.model.load_state_dict(ckpt, strict=False)
                missing = [k for k in missing if not k.startswith("fusion_gate.")]
                if missing or unexpected:
                    raise RuntimeError(
                        f"state_dict mismatch loading {self.cfg.pretrained_model_path}: "
                        f"missing={missing} unexpected={list(unexpected)}"
                    )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([{"name": "model", "params": self.model.parameters(), "lr": self.cfg.optimizer.lr}], weight_decay=self.cfg.optimizer.weight_decay, betas=(0.9, 0.98))
        scheduler = WarmupCosineScheduler(optimizer, self.cfg.optimizer.warmup_epochs, self.cfg.trainer.max_epochs, len(self.trainer.datamodule.train_dataloader()))
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def forward(self, video, audio):
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)
        video_feat, _ = self.model.encoder(video.unsqueeze(0).to(self.device), None)
        audio_feat, _ = self.model.aux_encoder(audio.unsqueeze(0).to(self.device), None)
        audiovisual_feat, _ = self.model.fuse(video_feat, audio_feat)

        audiovisual_feat = audiovisual_feat.squeeze(0)

        nbest_hyps = self.beam_search(audiovisual_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        return predicted

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="val")

    def test_step(self, sample, sample_idx):
        video_feat, _ = self.model.encoder(sample["video"].unsqueeze(0).to(self.device), None)
        audio_feat, _ = self.model.aux_encoder(sample["audio"].unsqueeze(0).to(self.device), None)
        audiovisual_feat, gate_weights = self.model.fuse(video_feat, audio_feat)

        # Accumulate the audio-stream gate weight over all frames. The test set
        # runs at a fixed decode.snr_target, so the epoch mean is one point on
        # the audio-weight-vs-SNR curve.
        self.total_audio_weight += gate_weights[..., 1].sum().item()
        self.total_frames += gate_weights.shape[1]

        audiovisual_feat = audiovisual_feat.squeeze(0)

        nbest_hyps = self.beam_search(audiovisual_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")

        token_id = sample["target"]
        actual = self.text_transform.post_process(token_id)

        if self.lang == "ja":
            self.total_edit_distance += compute_char_level_distance(actual, predicted)
            self.total_length += len(actual)
        else:
            self.total_edit_distance += compute_word_level_distance(actual, predicted)
            self.total_length += len(actual.split())
        return

    def _step(self, batch, batch_idx, step_type):
        loss, loss_ctc, loss_att, acc, audio_weight = self.model(batch["videos"], batch["audios"], batch["video_lengths"], batch["audio_lengths"], batch["targets"])
        batch_size = len(batch["videos"])

        if step_type == "train":
            self.log("loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("loss_ctc", loss_ctc, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("loss_att", loss_att, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("decoder_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size)
            # Mean gate weight on audio (0.5 = neutral). Rises for clean speech,
            # falls as training sees more low-SNR utterances shift trust to video.
            self.log("gate_w_audio", audio_weight, on_step=False, on_epoch=True, batch_size=batch_size)
        else:
            self.log("loss_val", loss, batch_size=batch_size)
            self.log("loss_ctc_val", loss_ctc, batch_size=batch_size)
            self.log("loss_att_val", loss_att, batch_size=batch_size)
            self.log("decoder_acc_val", acc, batch_size=batch_size)
            self.log("gate_w_audio_val", audio_weight, batch_size=batch_size)

        if step_type == "train":
            self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss

    def on_train_epoch_start(self):
        # Lightning 2.x: train_dataloader is the DataLoader itself (no .loaders).
        train_dl = self.trainer.train_dataloader
        sampler = getattr(train_dl, "batch_sampler", None) if train_dl is not None else None
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)
        return super().on_train_epoch_start()

    def on_test_epoch_start(self):
        self.total_length = 0
        self.total_edit_distance = 0
        self.total_audio_weight = 0.0
        self.total_frames = 0
        self.text_transform = get_text_transform(self.lang)
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)

    def on_test_epoch_end(self):
        # Japanese is scored at the character level (CER); other languages at the
        # word level (WER). Same value either way — only the metric label differs.
        metric_name = "cer" if self.lang == "ja" else "wer"
        self.log(metric_name, self.total_edit_distance / self.total_length)
        # Mean audio-stream gate weight at this run's decode.snr_target. Sweep
        # snr_target across eval runs to trace audio reliance vs. SNR.
        if self.total_frames > 0:
            self.log("gate_w_audio", self.total_audio_weight / self.total_frames)


def get_beam_search_decoder(model, token_list, ctc_weight=0.1, beam_size=40):
    scorers = {
        "decoder": model.decoder,
        "ctc": CTCPrefixScorer(model.ctc, model.eos),
        "length_bonus": LengthBonus(len(token_list)),
        "lm": None
    }

    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": 0.0,
        "length_bonus": 0.0,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )
