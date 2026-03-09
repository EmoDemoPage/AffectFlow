import os
import transformers
import torch
from torch import nn
from models.vocoder.vocos import (
    VocosBackbone,
    ISTFTHead,
    MultiResolutionDiscriminator,
    MultiPeriodDiscriminator,
)
from models.vocoder.vocos.loss import (
    DiscriminatorLoss,
    GeneratorLoss,
    FeatureMatchingLoss,
    MelSpecReconstructionLoss,
)
from utils.audio.io import save_wav
from utils.commons.hparams import hparams
from utils.nn.model_utils import print_arch
from tasks.vocoder.vocoder_base import VocoderBaseTask
from utils.commons.tensor_utils import tensors_to_scalars


class VocosTask(VocoderBaseTask):
    def build_model(self):
        self.model_gen = VocosBackbone(
            self.hparams["input_channels"],
            self.hparams["dim"],
            self.hparams["intermediate_dim"],
            self.hparams["num_layers"],
        )
        self.head = ISTFTHead(
            self.hparams["dim"],
            self.hparams["n_fft"],
            self.hparams["hop_length"],
            self.hparams["padding"],
        )

        self.model_disc = nn.ModuleDict()
        self.model_disc["mpd"] = MultiPeriodDiscriminator()
        self.model_disc["mrd"] = MultiResolutionDiscriminator()
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.melspec_loss = MelSpecReconstructionLoss(
            sample_rate=self.hparams["audio_sample_rate"]
        )

        print_arch(self.model_gen)
        if hparams["load_ckpt"] != "":
            self.load_ckpt(
                hparams["load_ckpt"], "model_gen", "model_gen", force=True, strict=True
            )
            self.load_ckpt(
                hparams["load_ckpt"],
                "model_disc",
                "model_disc",
                force=True,
                strict=True,
            )
        return self.model_gen

    def _training_step(self, sample, batch_idx, optimizer_idx):
        y = sample["wavs"].squeeze(1)
        x = sample["mels"]
        loss_output = {}
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################

            x = self.model_gen(x)
            y_ = self.head(x)
            self.y_ = y_.detach()
            _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.model_disc["mpd"](
                y=y,
                y_hat=y_,
            )
            _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.model_disc["mrd"](
                y=y,
                y_hat=y_,
            )
            loss_gen_mp, list_loss_gen_mp = self.gen_loss(disc_outputs=gen_score_mp)
            loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(disc_outputs=gen_score_mrd)
            loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)
            loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
            loss_fm_mp = self.feat_matching_loss(
                fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp
            ) / len(fmap_rs_mp)
            loss_fm_mrd = self.feat_matching_loss(
                fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd
            ) / len(fmap_rs_mrd)
            mel_loss = self.melspec_loss(y, y_)
            loss_output["loss_fm_mrd"] = loss_fm_mrd * self.hparams["mrd_loss_coeff"]
            loss_output["loss_fm_mp"] = loss_fm_mp
            loss_output["loss_gen_mrd"] = loss_gen_mrd * self.hparams["mrd_loss_coeff"]
            loss_output["loss_gen_mp"] = loss_gen_mp
            loss_output["mel_loss"] = mel_loss * self.hparams["mel_loss_coeff"]

        else:
            #######################
            #    Discriminator    #
            #######################
            y_ = self.y_
            # MPD
            real_score_mp, gen_score_mp, _, _ = self.model_disc["mpd"](y, y_.detach())
            loss_mp, loss_mp_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mp, disc_generated_outputs=gen_score_mp
            )
            # MSD
            real_score_mrd, gen_score_mrd, _, _ = self.model_disc["mrd"](y, y_.detach())
            loss_mrd, loss_mrd_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd
            )
            loss_mp /= len(loss_mp_real)
            loss_mrd /= len(loss_mrd_real)
            loss_output["loss_mp"] = loss_mp
            loss_output["loss_mrd"] = self.hparams["mrd_loss_coeff"] * loss_mrd

        total_loss = sum(loss_output.values())
        return total_loss, loss_output

    def validation_step(self, sample, batch_idx):
        outputs = {}

        if self.global_step % hparams["valid_infer_interval"] == 0 and batch_idx < 10:
            total_loss, loss_output = self._training_step(sample, batch_idx, 0)
            outputs["losses"] = tensors_to_scalars(loss_output)
            outputs["total_loss"] = tensors_to_scalars(total_loss)

            x = sample["mels"]
            y = sample["wavs"].squeeze(1)
            x = self.model_gen(x)
            y_ = self.head(x)

            for idx, (wav_pred, wav_gt, item_name) in enumerate(
                zip(y_, y, sample["item_name"])
            ):
                wav_pred = wav_pred / wav_pred.abs().max()
                if self.global_step == 0:
                    wav_gt = wav_gt / wav_gt.abs().max()
                    self.logger.add_audio(
                        f"wav_gt_{batch_idx}",
                        wav_gt,
                        self.global_step,
                        hparams["audio_sample_rate"],
                    )
                self.logger.add_audio(
                    f"wav_pred_{batch_idx}",
                    wav_pred,
                    self.global_step,
                    hparams["audio_sample_rate"],
                )
        return outputs

    def build_optimizer(self, model):
        optimizer_gen = torch.optim.AdamW(
            self.model_gen.parameters(),
            lr=hparams["initial_learning_rate"],
            betas=[hparams["adam_b1"], hparams["adam_b2"]],
        )
        optimizer_disc = torch.optim.AdamW(
            self.model_disc.parameters(),
            lr=hparams["initial_learning_rate"],
            betas=[hparams["adam_b1"], hparams["adam_b2"]],
        )
        return [optimizer_gen, optimizer_disc]

    def build_scheduler(self, optimizer):
        return {
            "gen": transformers.get_cosine_schedule_with_warmup(
                optimizer[0],
                num_warmup_steps=self.hparams["num_warmup_steps"],
                num_training_steps=self.hparams["max_steps"],
            ),
            "disc": transformers.get_cosine_schedule_with_warmup(
                optimizer[1],
                num_warmup_steps=self.hparams["num_warmup_steps"],
                num_training_steps=self.hparams["max_steps"],
            ),
        }

    def test_step(self, sample, batch_idx):
        loss_output = {}
        mels = sample["mels"]
        y = sample["wavs"].squeeze(1)

        x = self.model_gen(mels)
        y_ = self.head(x)
        print(y.shape, y_.shape, mels.shape)
        gen_dir = os.path.join(
            hparams["work_dir"],
            f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}',
        )
        os.makedirs(gen_dir, exist_ok=True)
        for idx, (wav_pred, wav_gt, item_name) in enumerate(
            zip(y_, y, sample["item_name"])
        ):
            wav_pred = wav_pred.clamp(-1, 1)
            save_wav(
                wav_pred.view(-1).cpu().float().numpy(),
                f"{gen_dir}/{item_name}.wav",
                hparams["audio_sample_rate"],
            )
        return loss_output
