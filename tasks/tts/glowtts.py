import math
import torch
from tasks.tts.speech_base import SpeechBaseTask
from utils.commons.tensor_utils import tensors_to_scalars
from utils.nn.schedulers import GlowTTSSchedule


class GlowTTSTask(SpeechBaseTask):

    def forward(self, sample, infer=False, *args, **kwargs):
        x = sample["txt_tokens"]  # [B, T_t]
        x_lengths = sample["txt_lengths"]
        y = sample["mels"]  # [B, T_s, 80]
        y_lengths = sample["mel_lengths"]  # [B, T_s, 80]
        spk_embed = sample.get("spk_embed")
        spk_id = sample.get("spk_ids")

        if not infer:
            output = self.model(x, x_lengths, y, y_lengths, gen=infer)
            # (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_)
            losses = {}

            losses["mle"] = self.mle_loss(
                output["z"],
                output["z_m"],
                output["z_logs"],
                output["logdet"],
                output["z_mask"],
            )
            losses["dur"] = self.duration_loss(
                output["logw"],
                output["logw_"],
                x_lengths,
            )
            return losses, output
        else:
            noise_scale = 0.667
            length_scale = 1.0
            output = self.model(
                x,
                x_lengths,
                gen=True,
                noise_scale=noise_scale,
                length_scale=length_scale,
            )
            return output

    def mle_loss(self, z, m, logs, logdet, mask):
        l = torch.sum(logs) + 0.5 * torch.sum(
            torch.exp(-2 * logs) * ((z - m) ** 2)
        )  # neg normal likelihood w/o the constant term
        l = l - torch.sum(logdet)  # log jacobian determinant
        l = l / torch.sum(
            torch.ones_like(z) * mask
        )  # averaging across batch, channel and time axes
        l = l + 0.5 * math.log(2 * math.pi)  # add the remaining constant term
        return l

    def duration_loss(self, logw, logw_, lengths):
        l = torch.sum((logw - logw_) ** 2) / torch.sum(lengths)
        return l

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs["losses"] = {}
        outputs["losses"], _ = self(sample)
        outputs["nsamples"] = sample["nsamples"]

        if (
            self.global_step % self.hparams["valid_infer_interval"] == 0
            and batch_idx < self.hparams["num_valid_plots"]
        ):
            model_out = self(sample, infer=True)
            self.save_valid_result(sample, batch_idx, model_out)

        outputs = tensors_to_scalars(outputs)
        return outputs

    def save_valid_result(self, sample, batch_idx, model_out):
        sr = self.hparams["audio_sample_rate"]
        gt = sample["mels"]
        pred = model_out["mel_out"]
        attn = model_out["attn"].cpu().numpy()
        self.plot_mel(batch_idx, [gt[0], pred[0]], title=f"mel_{batch_idx}")
        self.logger.add_image(
            f"plot_attn_{batch_idx}", self.plot_alignment(attn[0]), self.global_step
        )

        wav_pred = self.vocoder.spec2wav(pred[0].cpu())
        self.logger.add_audio(f"wav_pred_{batch_idx}", wav_pred, self.global_step, sr)

        if self.global_step <= self.hparams["valid_infer_interval"]:
            wav_gt = self.vocoder.spec2wav(gt[0].cpu())
            self.logger.add_audio(f"wav_gt_{batch_idx}", wav_gt, self.global_step, sr)

    def build_scheduler(self, optimizer):
        return GlowTTSSchedule(
            optimizer,
            self.hparams["lr"],
            self.hparams["warmup_updates"],
            self.hparams["hidden_channels"],
        )
