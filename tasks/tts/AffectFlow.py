import numpy as np
import os
import torch
import torch.nn.functional as F
from tasks.tts.fastspeech import FastSpeechTask
from utils.plot.plot import spec_to_figure
from utils.commons.tensor_utils import tensors_to_scalars
from utils.audio.pitch.utils import denorm_f0
import torch.nn as nn

class ExpressiveFS2Task(FastSpeechTask):

    def forward(self, sample, infer=False, *args, **kwargs):
        hparams = self.hparams
        txt_tokens = sample["txt_tokens"]  # [B, T_t]
        ph2word = sample["ph2word"]
        
        word_VAD_pad = sample["word_VAD_pad"]  # word_VAD_pad: (B, W_len, 3)
        wordsum_VAD_pad = sample["wordsum_VAD_pad"] # wordsum_VAD_pad: (B, Wsum_len, 3)
        wordsum_VAD_mask = sample["wordsum_VAD_mask"] # wordsum_VAD_mask: (B, Wsum_len)
        wordsum_bert_emb = sample["wordsum_bert_emb"] # wordsum_bert_emb: (B, Wsum_len, 768)
        qmask_bert_emb = sample["qmask_bert_emb"] # qmask_bert_emb: (B, Wsum_len, 2)
        wordsum_mask = sample["wordsum_mask"] # wordsum_mask: (B, Wsum_len)
        
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
                ph2word=ph2word,
                target=target,
                spk_id=spk_id,
                word_VAD=word_VAD_pad,
                wordsum_bert_emb=wordsum_bert_emb,
                qmask_bert_emb=qmask_bert_emb,
                wordsum_mask=wordsum_mask,
                f0=f0,
                uv=uv,
                energy=energy,
                infer=False,
            )
            losses = {}
            ccc_loss_fn = MaskedCCCLoss(reduction="mean", per_batch=False)
            losses["word_ccc_loss"] = ccc_loss_fn(output["log_prob"], wordsum_VAD_pad, wordsum_VAD_mask)

            self.add_mel_loss(output["mel_out"], target, losses)
            self.add_dur_loss(output["dur"], mel2ph, txt_tokens, losses=losses)
            self.add_pitch_loss(output, sample, losses)
            self.add_energy_loss(output, sample, losses)
            
            return losses, output
        else:
            target = sample["mels"]  # [B, T_s, 80]
            output = self.model(
                txt_tokens,
                ph2word=ph2word,
                target=target,
                spk_id=spk_id,
                word_VAD=word_VAD_pad,
                wordsum_bert_emb=wordsum_bert_emb,
                qmask_bert_emb=qmask_bert_emb,
                wordsum_mask=wordsum_mask,
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

    def test_step(self, sample, batch_idx):
        """
        :param sample:
        :param batch_idx:
        :return:
        """
        assert (
            sample["txt_tokens"].shape[0] == 1
        ), "only support batch_size=1 in inference"
        outputs = self(sample, infer=True)
        text = sample["text"][0]
        item_name = sample["item_name"][0]
        tokens = sample["txt_tokens"][0].cpu().numpy()
        mel_gt = sample["mels"][0].cpu().numpy()
        mel_pred = outputs["mel_out"][0].cpu().numpy()
        vad_pred = outputs["word_vad_pred"].detach().cpu().numpy().squeeze(0)  # [W,3]
        vad_gt   = sample["word_VAD_pad"].detach().cpu().numpy().squeeze(0)    # [W,3]
        vad_mask = sample["word_VAD_mask"].detach().cpu().numpy().squeeze(0)   # [W] or [W,3]

        mel2ph_item = sample.get("mel2ph")
        if mel2ph_item is not None:
            mel2ph = mel2ph_item[0].cpu().numpy()
        else:
            mel2ph = None
        mel2ph_pred_item = outputs.get("mel2ph")
        if mel2ph_pred_item is not None:
            mel2ph_pred = mel2ph_pred_item[0].cpu().numpy()
        else:
            mel2ph_pred = None
        str_phs = self.token_encoder.decode(tokens, strip_padding=True)

        base_fn = item_name
        gen_dir = self.gen_dir
        wav_pred = self.vocoder.spec2wav(mel_pred)

        audio_sample_rate = self.hparams["audio_sample_rate"]
        out_wav_norm = (self.hparams["out_wav_norm"],)
        mel_vmin = self.hparams["mel_vmin"]
        mel_vmax = self.hparams["mel_vmax"]
        save_mel_npy = self.hparams["save_mel_npy"]

        self.saving_result_pool.add_job(
            self.save_result,
            args=[
                wav_pred,
                mel_pred,
                base_fn,
                gen_dir,
                vad_pred,
                vad_gt,
                vad_mask,
                str_phs,
                mel2ph_pred,
                None,
                audio_sample_rate,
                out_wav_norm,
                mel_vmin,
                mel_vmax,
                save_mel_npy,
            ],
        )
        if self.hparams["save_gt"]:
            gt_name = base_fn + "_gt"
            wav_gt = self.vocoder.spec2wav(mel_gt)
            self.saving_result_pool.add_job(
                self.save_result,
                args=[
                    wav_gt,
                    mel_gt,
                    gt_name,
                    gen_dir,
                    vad_pred,
                    vad_gt,
                    vad_mask,
                    str_phs,
                    mel2ph,
                    None,
                    audio_sample_rate,
                    out_wav_norm,
                    mel_vmin,
                    mel_vmax,
                    save_mel_npy,
                ],
            )

        return {
            "item_name": item_name,
            "text": text,
            "ph_tokens": self.token_encoder.decode(tokens.tolist()),
            "wav_fn_pred": base_fn,
            "wav_fn_gt": base_fn + "_gt",
        }

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

class MaskedCCCLoss(nn.Module):
    """
    Masked CCC loss for word-level VAD.

    pred: (B, W, C)  C=3
    lab : (B, W, C)
    mask: (B, W) or (B, W, 1) or (B, W, C)  values in {0,1}
    returns: scalar loss by default (1 - mean_ccc)
    """
    def __init__(self, eps: float = 1e-8, reduction: str = "mean", per_batch: bool = False):
        super().__init__()
        assert reduction in ["mean", "none"]
        self.eps = eps
        self.reduction = reduction
        self.per_batch = per_batch  # if True: compute CCC per sample then average; else flatten all valid words

    def forward(self, pred: torch.Tensor, lab: torch.Tensor, mask: torch.Tensor):
        # pred/lab: [B, W, C]
        assert pred.shape == lab.shape, f"pred {pred.shape} != lab {lab.shape}"
        assert pred.dim() == 3, "pred must be (B, W, C)"

        B, W, C = pred.shape

        # Normalize mask to [B, W]
        if mask.dim() == 3:
            # [B, W, 1] or [B, W, C] -> [B, W]
            mask2d = mask[..., 0]
        elif mask.dim() == 2:
            mask2d = mask
        else:
            raise ValueError(f"mask must be 2D or 3D, got {mask.shape}")

        mask2d = mask2d.to(dtype=pred.dtype)  # float mask
        mask2d = (mask2d > 0.5).to(dtype=pred.dtype)  # ensure 0/1

        if self.per_batch:
            # CCC computed per sample over its valid words, then average
            ccc_list = []
            for b in range(B):
                mb = mask2d[b]  # [W]
                idx = mb.nonzero(as_tuple=False).squeeze(-1)  # [Nb]
                if idx.numel() < 2:
                    # not enough points to compute stable CCC -> ignore (treat as 0 contribution)
                    ccc_b = pred.new_zeros(C)
                else:
                    pb = pred[b, idx, :]  # [Nb, C]
                    lb = lab[b, idx, :]   # [Nb, C]
                    ccc_b = self._ccc(pb, lb)  # [C]
                ccc_list.append(ccc_b)

            ccc = torch.stack(ccc_list, dim=0)  # [B, C]
            # loss = 1 - ccc
            loss = 1.0 - ccc
            return loss.mean() if self.reduction == "mean" else loss

        else:
            # Flatten all valid words across batch
            valid = mask2d.bool()  # [B, W]
            if valid.sum() < 2:
                # nothing (or too little) valid -> return 0 loss (or 1? you decide)
                return pred.new_zeros(()) if self.reduction == "mean" else pred.new_zeros(B, C)

            p = pred[valid]  # [N, C]
            l = lab[valid]   # [N, C]
            ccc = self._ccc(p, l)  # [C]
            loss = 1.0 - ccc       # [C]
            return loss.mean() if self.reduction == "mean" else loss

    def _ccc(self, pred_nc: torch.Tensor, lab_nc: torch.Tensor):
        """
        pred_nc: (N, C)
        lab_nc : (N, C)
        returns: (C,) ccc per channel
        """
        # Means
        m_pred = pred_nc.mean(dim=0, keepdim=True)  # (1, C)
        m_lab  = lab_nc.mean(dim=0, keepdim=True)   # (1, C)

        # Demean
        d_pred = pred_nc - m_pred  # (N, C)
        d_lab  = lab_nc - m_lab

        # Var / Std (population)
        v_pred = (d_pred ** 2).mean(dim=0)  # (C,)
        v_lab  = (d_lab ** 2).mean(dim=0)   # (C,)
        s_pred = torch.sqrt(v_pred + self.eps)
        s_lab  = torch.sqrt(v_lab + self.eps)

        # Corr: cov / (std*std)
        cov = (d_pred * d_lab).mean(dim=0)  # (C,)
        corr = cov / (s_pred * s_lab + self.eps)

        # CCC
        mean_diff_sq = (m_pred.squeeze(0) - m_lab.squeeze(0)) ** 2  # (C,)
        ccc = (2.0 * corr * s_pred * s_lab) / (v_pred + v_lab + mean_diff_sq + self.eps)  # (C,)
        return ccc
