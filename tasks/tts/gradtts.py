from torch.nn import functional as F
from tasks.tts.speech_base import SpeechBaseTask
from utils.commons.tensor_utils import tensors_to_scalars


class GradTTSTask(SpeechBaseTask):

    def forward(self, sample, infer=False, *args, **kwargs):
        x = sample["txt_tokens"]  # [B, T_t]
        x_lengths = sample["txt_lengths"]
        y = sample["mels"]  # [B, T_s, 80]
        y_lengths = sample["mel_lengths"]  # [B, T_s, 80]
        spk_embed = sample.get("spk_embed")
        spk_id = sample.get("spk_ids")

        if not infer:
            output = self.model.compute_loss(
                x,
                x_lengths,
                y.transpose(1, 2),
                y_lengths,
                spk=spk_id,
                out_size=self.hparams["out_size"],
            )
            losses = {}
            losses["dur_loss"] = output["dur_loss"]
            losses["prior_loss"] = output["prior_loss"]
            losses["diff_loss"] = output["diff_loss"]

            return losses, output
        else:

            output = self.model(x, x_lengths, spk=spk_id, n_timesteps=50)
            return output

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
        prior = model_out["encoder_outputs"]
        attn = model_out["attn"].cpu().numpy()

        self.plot_mel(batch_idx, [gt[0], prior[0], pred[0]], title=f"mel_{batch_idx}")
        self.logger.add_image(
            f"plot_attn_{batch_idx}", self.plot_alignment(attn[0]), self.global_step
        )

        wav_pred = self.vocoder.spec2wav(pred[0].cpu())
        self.logger.add_audio(f"wav_pred_{batch_idx}", wav_pred, self.global_step, sr)

        wav_pred = self.vocoder.spec2wav(prior[0].cpu())
        self.logger.add_audio(f"wav_prior_{batch_idx}", wav_pred, self.global_step, sr)

        if self.global_step <= self.hparams["valid_infer_interval"]:
            wav_gt = self.vocoder.spec2wav(gt[0].cpu())
            self.logger.add_audio(f"wav_gt_{batch_idx}", wav_gt, self.global_step, sr)
