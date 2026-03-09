# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import random

import torch

import monotonic_align
from models.tts.gradtts.base import BaseModule
from models.tts.gradtts.text_encoder import TextEncoder
from models.tts.gradtts.diffusion import Diffusion
from models.tts.gradtts.utils import (
    sequence_mask,
    generate_path,
    duration_loss,
    fix_len_compatibility,
)


class GradTTS(BaseModule):
    def __init__(self, dict_size, hparams, out_dims=80):
        super().__init__()
        self.n_vocab = dict_size
        self.n_spks = hparams["n_spks"]
        self.spk_emb_dim = hparams["spk_emb_dim"]
        self.n_enc_channels = hparams["n_enc_channels"]
        self.filter_channels = hparams["filter_channels"]
        self.filter_channels_dp = hparams["filter_channels_dp"]
        self.n_heads = hparams["n_heads"]
        self.n_enc_layers = hparams["n_enc_layers"]
        self.enc_kernel = hparams["enc_kernel"]
        self.enc_dropout = hparams["enc_dropout"]
        self.window_size = hparams["window_size"]
        self.n_feats = out_dims
        self.dec_dim = hparams["dec_dim"]
        self.beta_min = hparams["beta_min"]
        self.beta_max = hparams["beta_max"]
        self.pe_scale = hparams["pe_scale"]

        if self.n_spks > 1:
            self.spk_emb = torch.nn.Embedding(self.n_spks, self.spk_emb_dim)

        self.encoder = TextEncoder(
            self.n_vocab,
            self.n_feats,
            self.n_enc_channels,
            self.filter_channels,
            self.filter_channels_dp,
            self.n_heads,
            self.n_enc_layers,
            self.enc_kernel,
            self.enc_dropout,
            self.window_size,
        )
        self.decoder = Diffusion(
            self.n_feats,
            self.dec_dim,
            self.n_spks,
            self.spk_emb_dim,
            self.beta_min,
            self.beta_max,
            self.pe_scale,
        )

    @torch.no_grad()
    def forward(
        self,
        x,
        x_lengths,
        n_timesteps,
        temperature=1.0,
        stoc=False,
        spk=None,
        length_scale=1.0,
    ):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        x, x_lengths = self.relocate_input([x, x_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(
            attn.squeeze(1).transpose(1, 2).contiguous(),
            mu_x.transpose(1, 2).contiguous(),
        )
        mu_y = mu_y.transpose(1, 2).contiguous()
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps, stoc, spk)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]
        ret = {}
        ret["encoder_outputs"] = encoder_outputs.transpose(1, 2).contiguous()
        ret["mel_out"] = decoder_outputs.transpose(1, 2).contiguous()
        ret["attn"] = attn[:, :, :y_max_length].squeeze(1)

        return ret

    def compute_loss(self, x, x_lengths, y, y_lengths, spk=None, out_size=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad():
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2).contiguous(), y**2)
            y_mu_double = torch.matmul(
                2.0 * (factor * mu_x).transpose(1, 2).contiguous(), y
            )
            mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(
                zip([0] * max_offset.shape[0], max_offset.cpu().numpy())
            )
            out_offset = torch.LongTensor(
                [
                    torch.tensor(random.choice(range(start, end)) if end > start else 0)
                    for start, end in offset_ranges
                ]
            ).to(y_lengths)

            attn_cut = torch.zeros(
                attn.shape[0],
                attn.shape[1],
                out_size,
                dtype=attn.dtype,
                device=attn.device,
            )
            y_cut = torch.zeros(
                y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device
            )
            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(
            attn.squeeze(1).transpose(1, 2).contiguous(),
            mu_x.transpose(1, 2).contiguous(),
        )
        mu_y = mu_y.transpose(1, 2).contiguous()

        # Compute loss of score-based decoder
        diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y, spk)

        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        ret = {}
        ret["dur_loss"] = dur_loss
        ret["prior_loss"] = prior_loss
        ret["diff_loss"] = diff_loss

        return ret
