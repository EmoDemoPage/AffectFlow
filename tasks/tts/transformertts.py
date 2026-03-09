import torch.nn.functional as F
import matplotlib.pyplot as plt
from tasks.tts.speech_base import SpeechBaseTask
from utils.commons.tensor_utils import tensors_to_scalars


class TransformerTTSTask(SpeechBaseTask):
    def __init__(self):
        super().__init__()

    def forward(self, sample, infer=False, *args, **kwargs):
        txt_tokens = sample["txt_tokens"]  # [B, T_t]
        txt_lengths = sample["txt_lengths"]

        spk_embed = sample.get("spk_embed")
        spk_id = sample.get("spk_ids")

        if not infer:
            mels = sample["mels"]
            mel_lengths = sample["mel_lengths"]
            stop = sample["gates"]
            output = self.model(txt_tokens, txt_lengths, mels, mel_lengths)

            # mel_loss, bce_loss, guide_loss = criterion(
            #     (mel_out, mel_out_post, gate_out),
            #     (melspec, gate),
            #     (enc_dec_alignments, text_lengths, mel_lengths),
            # )

            mel_out = output["mel_out"]
            mel_out_post = output["mel_out_post"]
            stop_preds = output["gate_out"]

            losses = {}
            mel_loss = F.l1_loss(mel_out, mels) + F.l1_loss(mel_out_post, mels)
            stop_loss = F.binary_cross_entropy_with_logits(stop_preds, stop)

            losses["mel"] = mel_loss
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
        self.plot_mel(
            batch_idx,
            [sample["mels"][0], mel_out[0]],
            name=f"mel_{batch_idx}",
            title=f"mel_{batch_idx}",
        )

        # text = sample["txt_tokens"]
        # text_lengths = sample["txt_lengths"]

        # mel_lengths = sample["mel_lengths"]
        # attn = model_out["enc_dec_alignments"].cpu()
        # enc_dec_align_fig = self.plot_alignments(
        #     attn, text, mel_lengths, text_lengths, "enc_dec"
        # )
        # self.logger.add_figure(
        #     f"plot_attn_{batch_idx}", enc_dec_align_fig, self.global_step
        # )

        wav_pred = self.vocoder.spec2wav(mel_out[0].cpu())
        self.logger.add_audio(f"wav_pred_{batch_idx}", wav_pred, self.global_step, sr)

        # gt wav
        if self.global_step <= self.hparams["valid_infer_interval"]:
            mel_gt = sample["mels"][0].cpu()
            wav_gt = self.vocoder.spec2wav(mel_gt)
            self.logger.add_audio(f"wav_gt_{batch_idx}", wav_gt, self.global_step, sr)

    def plot_alignments(self, alignments, text, mel_lengths, text_lengths, att_type):
        fig, axes = plt.subplots(
            self.hparams.n_layers,
            self.hparams.n_heads,
            figsize=(5 * self.hparams.n_heads, 5 * self.hparams.n_layers),
        )

        L, T = text_lengths[-1], mel_lengths[-1]
        n_layers, n_heads = alignments.size(0), alignments.size(1)

        for layer in range(n_layers):
            for head in range(n_heads):
                if att_type == "enc":
                    align = alignments[-1, layer, head].contiguous()
                    axes[layer, head].imshow(align[:L, :L], aspect="auto")
                    axes[layer, head].xaxis.tick_top()

                elif att_type == "dec":
                    align = alignments[-1, layer, head].contiguous()
                    axes[layer, head].imshow(align[:T, :T], aspect="auto")
                    axes[layer, head].xaxis.tick_top()

                elif att_type == "enc_dec":

                    align = alignments[layer, head].transpose(0, 1).contiguous()
                    axes[layer, head].imshow(
                        align[:L, :T], origin="lower", aspect="auto"
                    )
        plt.tight_layout()
        return fig
