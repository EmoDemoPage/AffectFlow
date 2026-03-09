import torch
from torch import nn

from models.commons.layers import LayerNorm
import torch.nn.functional as F


class DurationPredictor(torch.nn.Module):
    def __init__(
        self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0
    ):
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        self.linear = nn.Sequential(torch.nn.Linear(n_chans, 1), nn.Softplus())

    def forward(self, x, x_padding=None):
        x = x.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            x = f(x)  # (B, C, Tmax)
            if x_padding is not None:
                x = x * (1 - x_padding.float())[:, None, :]

        x = self.linear(x.transpose(1, -1))  # [B, T, C]
        x = x * (1 - x_padding.float())[:, :, None]  # (B, T, C)
        x = x[..., 0]  # (B, Tmax)
        return x


class LengthRegulator(torch.nn.Module):
    def __init__(self, pad_value=0.0):
        super(LengthRegulator, self).__init__()
        self.pad_value = pad_value

    def forward(self, dur, dur_padding=None, alpha=1.0):
        """
        Example (no batch dim version):
            1. dur = [2,2,3]
            2. token_idx = [[1],[2],[3]], dur_cumsum = [2,4,7], dur_cumsum_prev = [0,2,4]
            3. token_mask = [[1,1,0,0,0,0,0],
                             [0,0,1,1,0,0,0],
                             [0,0,0,0,1,1,1]]
            4. token_idx * token_mask = [[1,1,0,0,0,0,0],
                                         [0,0,2,2,0,0,0],
                                         [0,0,0,0,3,3,3]]
            5. (token_idx * token_mask).sum(0) = [1,1,2,2,3,3,3]

        :param dur: Batch of durations of each frame (B, T_txt)
        :param dur_padding: Batch of padding of each frame (B, T_txt)
        :param alpha: duration rescale coefficient
        :return:
            mel2ph (B, T_speech)
        assert alpha > 0
        """
        dur = torch.round(dur.float() * alpha).long()
        if dur_padding is not None:
            dur = dur * (1 - dur_padding.long())
        token_idx = torch.arange(1, dur.shape[1] + 1)[None, :, None].to(dur.device)
        dur_cumsum = torch.cumsum(dur, 1)
        dur_cumsum_prev = F.pad(dur_cumsum, [1, -1], mode="constant", value=0)

        pos_idx = torch.arange(dur.sum(-1).max())[None, None].to(dur.device)
        token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (
            pos_idx < dur_cumsum[:, :, None]
        )
        mel2token = (token_idx * token_mask.long()).sum(1)
        return mel2token


class PitchPredictor(torch.nn.Module):
    def __init__(
        self, idim, n_layers=5, n_chans=384, odim=2, kernel_size=5, dropout_rate=0.1
    ):
        super(PitchPredictor, self).__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans, n_chans, kernel_size, padding=kernel_size // 2
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        self.linear = torch.nn.Linear(n_chans, odim)

    def forward(self, x):
        """

        :param x: [B, T, H]
        :return: [B, T, H]
        """
        x = x.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            x = f(x)  # (B, C, Tmax)
        x = self.linear(x.transpose(1, -1))  # (B, Tmax, H)
        return x


class EnergyPredictor(PitchPredictor):
    pass

class VADPredictor(nn.Module):
    def __init__(self, idim, n_layers=2, n_chans=256, odim=3, dropout_rate=0.1):
        super().__init__()
        self.odim = odim

        self.prev_vad_embed = nn.Linear(odim, n_chans)

        self.gru = nn.GRU(
            input_size=idim + n_chans,
            hidden_size=n_chans,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_rate if n_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(n_chans, odim)

    def forward(self, x, gt_vad=None, infer=False, padding=None):
        B, W, C = x.shape
        device = x.device

        if (not infer) and (gt_vad is not None):
            prev = x.new_zeros(B, W, self.odim)
            if W > 1:
                prev[:, 1:, :] = gt_vad[:, :-1, :]
            prev_emb = self.prev_vad_embed(prev)                   # [B, W, n_chans]
            gru_inp = torch.cat([x, prev_emb], dim=-1)             # [B, W, C+n_chans]
            h, _ = self.gru(gru_inp)                               # [B, W, n_chans]
            out = self.proj(h)                                     # [B, W, 3]
            out = torch.sigmoid(out)                               

        else:
            out_steps = []
            h_t = None
            prev_v = x.new_zeros(B, self.odim)                     # start token: 0
            for t in range(W):
                prev_emb = self.prev_vad_embed(prev_v)             # [B, n_chans]
                inp_t = torch.cat([x[:, t, :], prev_emb], dim=-1)  # [B, C+n_chans]
                inp_t = inp_t[:, None, :]                          # [B, 1, ...]
                y_t, h_t = self.gru(inp_t, h_t)                    # y_t: [B,1,n_chans]
                v_t = torch.sigmoid(self.proj(y_t[:, 0, :]))       # [B,3] 0~1
                out_steps.append(v_t)
                prev_v = v_t
            out = torch.stack(out_steps, dim=1)                    # [B, W, 3]

        if padding is not None:
            out = out * (~padding).float()[:, :, None]
        return out
