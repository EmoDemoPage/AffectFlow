import torch.nn.functional as F
from tasks.tts.speech_base import SpeechBaseTask
from utils.commons.tensor_utils import tensors_to_scalars


class Tacotron2Task(SpeechBaseTask):
    def __init__(self):
        super().__init__()

    def forward(self, sample, infer=False, *args, **kwargs):
        txt_tokens = sample["txt_tokens"]  # [B, T_t]3
        txt_lengths = sample["txt_lengths"]
        spk_embed = sample.get("spk_embed")
        spk_id = sample.get("spk_ids")

        if not infer:
            mels = sample["mels"]
            mel_lengths = sample["mel_lengths"]
            stop = sample["gates"]
            output = self.model(txt_tokens, txt_lengths, mels, mel_lengths)
            decoder_output = output["decoder_output"]
            mel_out = output["mel_out"]
            stop_preds = output["stop_preds"]

            losses = {}
            mel_loss = F.mse_loss(mel_out, mels) + F.mse_loss(decoder_output, mels)
            stop_loss = F.binary_cross_entropy_with_logits(stop_preds, stop)
            losses["mel_loss"] = mel_loss
            losses["stop_loss"] = stop_loss * self.hparams["lambda_stop"]
            return losses, output
        else:
            output = self.model.inference(txt_tokens)
            return output

    def validation_step(self, sample, batch_idx):
        outputs = {}

        if (
            self.global_step % self.hparams["valid_infer_interval"] == 0
            and batch_idx < self.hparams["num_valid_plots"]
        ):
            outputs["losses"] = {}
            model_out = self(sample, infer=True)
            outputs["total_loss"] = sum(outputs["losses"].values())
            outputs["nsamples"] = sample["nsamples"]
            outputs = tensors_to_scalars(outputs)
            self.save_valid_result(sample, batch_idx, model_out)
        return outputs

    def save_valid_result(self, sample, batch_idx, model_out):
        sr = self.hparams["audio_sample_rate"]
        mel_out = model_out["mel_out"]
        attn = model_out["alignments"].cpu().numpy()
        self.plot_mel(
            batch_idx,
            [sample["mels"][0], mel_out[0]],
            name=f"mel_{batch_idx}",
            title=f"mel_{batch_idx}",
        )
        self.logger.add_image(
            f"plot_attn_{batch_idx}", self.plot_alignment(attn[0]), self.global_step
        )

        wav_pred = self.vocoder.spec2wav(mel_out[0].cpu())
        self.logger.add_audio(f"wav_pred_{batch_idx}", wav_pred, self.global_step, sr)

        # gt wav
        if self.global_step <= self.hparams["valid_infer_interval"]:
            mel_gt = sample["mels"][0].cpu()
            wav_gt = self.vocoder.spec2wav(mel_gt)
            self.logger.add_audio(f"wav_gt_{batch_idx}", wav_gt, self.global_step, sr)
