import numpy as np
import torch
import torch.nn.functional as F
from tasks.tts.fastspeech import FastSpeechTask
from utils.plot.plot import spec_to_figure
from utils.commons.tensor_utils import tensors_to_scalars


class FastSpeech2Task(FastSpeechTask):

    def forward(self, sample, infer=False, *args, **kwargs):
        txt_tokens = sample["txt_tokens"]  # [B, T_t]
        spk_embed = sample.get("spk_embed")
        spk_id = sample.get("spk_ids")
        if not infer:
            target = sample["mels"]  # [B, T_s, 80]
            mel2ph = sample["mel2ph"]  # [B, T_s]
            f0 = sample.get("f0")
            uv = sample.get("uv")
            energy = sample.get("energy")
            output = self.model(
                txt_tokens,
                mel2ph=mel2ph,
                spk_embed=spk_embed,
                spk_id=spk_id,
                f0=f0,
                uv=uv,
                energy=energy,
                infer=False,
            )
            losses = {}
            self.add_mel_loss(output["mel_out"], target, losses)
            self.add_dur_loss(output["dur"], mel2ph, txt_tokens, losses=losses)
            self.add_pitch_loss(output, sample, losses)
            self.add_energy_loss(output, sample, losses)
            return losses, output
        else:

            output = self.model(
                txt_tokens,
                spk_embed=spk_embed,
                spk_id=spk_id,
                infer=True,
            )
            return output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs["losses"] = {}
        outputs["losses"], model_out = self(sample)
        outputs["total_loss"] = sum(outputs["losses"].values())
        outputs["nsamples"] = sample["nsamples"]
        outputs = tensors_to_scalars(outputs)
        if (
            self.global_step % self.hparams["valid_infer_interval"] == 0
            and batch_idx < self.hparams["num_valid_plots"]
        ):
            self.plot_cwt(batch_idx, model_out["cwt"], sample["cwt_spec"])
            model_out = self(sample, infer=True)
            self.save_valid_result(sample, batch_idx, model_out)
        return outputs

    def plot_cwt(self, batch_idx, cwt_out, cwt_gt=None):
        if len(cwt_out.shape) == 3:
            cwt_out = cwt_out[0]
        if isinstance(cwt_out, torch.Tensor):
            cwt_out = cwt_out.cpu().numpy()

        if len(cwt_gt.shape) == 3:
            cwt_gt = cwt_gt[0]
        if isinstance(cwt_gt, torch.Tensor):
            cwt_gt = cwt_gt.cpu().numpy()
        cwt_out = np.concatenate([cwt_gt, cwt_out], -1)
        name = f"plot_cwt_{batch_idx}"
        self.logger.add_figure(name, spec_to_figure(cwt_out), self.global_step)

    def add_pitch_loss(self, output, sample, losses):
        cwt_spec = sample[f"cwt_spec"]
        f0_mean = sample["f0_mean"]
        uv = sample["uv"]
        mel2ph = sample["mel2ph"]
        f0_std = sample["f0_std"]
        cwt_pred = output["cwt"][:, :, :10]
        f0_mean_pred = output["f0_mean"]
        f0_std_pred = output["f0_std"]
        nonpadding = (mel2ph != 0).float()
        losses["f0_cwt"] = F.l1_loss(cwt_pred, cwt_spec) * self.hparams["lambda_f0"]

        assert output["cwt"].shape[-1] == 11
        uv_pred = output["cwt"][:, :, -1]
        losses["uv"] = (
            (
                F.binary_cross_entropy_with_logits(uv_pred, uv, reduction="none")
                * nonpadding
            ).sum()
            / nonpadding.sum()
            * self.hparams["lambda_uv"]
        )
        losses["f0_mean"] = F.l1_loss(f0_mean_pred, f0_mean) * self.hparams["lambda_f0"]
        losses["f0_std"] = F.l1_loss(f0_std_pred, f0_std) * self.hparams["lambda_f0"]

    def add_energy_loss(self, output, sample, losses):
        energy_pred, energy = output["energy_pred"], sample["energy"]
        nonpadding = (energy != 0).float()
        loss = (
            F.mse_loss(energy_pred, energy, reduction="none") * nonpadding
        ).sum() / nonpadding.sum()
        loss = loss * self.hparams["lambda_energy"]
        losses["energy"] = loss
