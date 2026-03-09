import torch
from torch import nn
from models.commons.layers import Embedding
from models.commons.nar_tts_modules import EnergyPredictor, PitchPredictor, VADPredictor
from models.tts.commons.align_ops import expand_states, mel2ph_to_mel2word, word_vad_to_frame_vad, word_vad_to_frame_vad_simple
from models.tts.fastspeech import FastSpeech
from models.tts.AffectFlow.dialoguernn import BiModel, MatchingAttention
from utils.audio.cwt import cwt2f0, get_lf0_cwt
from utils.audio.pitch.utils import denorm_f0, f0_to_coarse, norm_f0
import numpy as np
import torch.nn.functional as F

class ExpressiveFS2(FastSpeech):
    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__(dict_size, hparams, out_dims)
        self.pitch_embed = Embedding(300, self.hidden_size, 0)
        self.energy_embed = Embedding(300, self.hidden_size, 0)
        self.energy_predictor = EnergyPredictor(
            self.hidden_size,
            n_chans=self.hidden_size,
            n_layers=hparams["predictor_layers"],
            dropout_rate=hparams["predictor_dropout"],
            odim=2,
            kernel_size=hparams["predictor_kernel"],
        )
        self.pitch_predictor = PitchPredictor(
            self.hidden_size,
            n_chans=self.hidden_size,
            n_layers=hparams["predictor_layers"],
            dropout_rate=hparams["predictor_dropout"],
            odim=11,
            kernel_size=hparams["predictor_kernel"],
        )
        self.cwt_stats_layers = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2),
        )
        self.vad_encoder = nn.Sequential(
            nn.Linear(3, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.spk_id_proj = Embedding(hparams["num_spk"], self.hidden_size)

        self.dialoguernn = BiModel(D_m = 100,  # dimension of utterance embeddings
                        D_g = 500,  # dimension of global states
                        D_p = 500,  # dimension of party states
                        D_e = 300,  # dimension of emotion states
                        D_h = 300,  # dimension of linear hidden states
                        n_classes=3,
                        listener_state=False,
                        context_attention="general",
                        dropout_rec=0.1,
                        dropout=0.1)
        
        self.vad_encoder = nn.Sequential(
            nn.Linear(3, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        
        self.wordsum_bert_in_proj = nn.Linear(768, 100)

    def forward(
        self,
        txt_tokens,
        mel2ph=None,
        ph2word=None,
        spk_embed=None,
        spk_id=None,
        word_VAD=None, # word_VAD: (B, W_len, 3)
        wordsum_bert_emb=None, # utterence_bert_emb: (B, U_len, 768)
        qmask_bert_emb=None, # qmask_bert_emb: (B, U_len, 2)
        wordsum_mask=None, # utt_mask: (B, U_len)
        f0=None,
        uv=None,
        energy=None,
        infer=False,
        **kwargs
    ):
        ret = {}
        encoder_out = self.encoder(txt_tokens)  # [B, T, C]
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]

        spk_embed = 0
        spk_embed = spk_embed + self.spk_id_proj(spk_id)

        style_embed = (spk_embed)[:, None, :]

        # add dur
        dur_inp = (encoder_out + style_embed) * src_nonpadding
        mel2ph = self.forward_dur(dur_inp, mel2ph, txt_tokens, ret, infer)
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = decoder_inp_ = expand_states(encoder_out, mel2ph)

        # utterence dialoguernn
        wordsum_bert_emb = self.wordsum_bert_in_proj(wordsum_bert_emb) # (B, U_len, 100)
        ret["log_prob"], _, _, _, _ = self.dialoguernn(wordsum_bert_emb.transpose(0,1), qmask_bert_emb.transpose(0,1), wordsum_mask, att2=True) # log_prob: (batch, Wsum_max, 3)
        pred_mel2word = mel2ph_to_mel2word(mel2ph, ph2word)
        
        if infer:
            log_prob = ret["log_prob"]              # (1, Wsum, 3)
            W_need = int(word_VAD.size(1))

            ret["word_vad_pred"] = word_vad_for_embed = log_prob[:, -W_need:, :]   # (1, W_need, 3)

        else:
            word_vad_for_embed = word_VAD

        # --------------------------
        # (B) word_vad -> frame_vad -> embed
        # --------------------------
        pred_frame_VAD = word_vad_to_frame_vad_simple(word_vad_for_embed, pred_mel2word)
        VAD_embed = self.vad_encoder(pred_frame_VAD)


        # add pitch and energy embed
        pitch_inp = (decoder_inp_ + style_embed + VAD_embed) * tgt_nonpadding
        decoder_inp = decoder_inp + self.forward_pitch(
            pitch_inp, f0, uv, mel2ph, ret, infer
        )

        # add pitch and energy embed
        energy_inp = (decoder_inp_ + style_embed + VAD_embed) * tgt_nonpadding
        decoder_inp = decoder_inp + self.forward_energy(energy_inp, energy, ret, infer)

        # decoder input
        decoder_inp = decoder_inp + style_embed + VAD_embed
        
        ret["mel_out"] = self.forward_decoder(
            decoder_inp, tgt_nonpadding, ret, infer=infer, **kwargs
        )
        return ret

    def _last_span_from_qmask(qmask: torch.Tensor):
        """
        qmask: (B, T, 2), pad=[0,0], real=[1,0] or [0,1]
        return: start,end (python slice), both (B,)
        """
        B, T, _ = qmask.shape
        is_pad = (qmask.abs().sum(dim=-1) == 0)         # (B,T)
        nonpad_len = (~is_pad).long().sum(dim=1)        # (B,)
        end = nonpad_len.clone()                         # exclusive

        start = torch.zeros_like(end)
        for b in range(B):
            e = int(end[b].item())
            if e == 0:
                start[b] = 0
                continue
            last_spk = qmask[b, e-1, :]                 # (2,)
            s = e - 1
            while s >= 0:
                if is_pad[b, s]:
                    break
                if not torch.equal(qmask[b, s, :], last_spk):
                    break
                s -= 1
            start[b] = s + 1
        return start, end


    def forward_pitch(self, decoder_inp, f0, uv, mel2ph, ret, infer=False):
        pitch_padding = mel2ph == 0
        ret["cwt"] = cwt_out = self.pitch_predictor(decoder_inp)
        stats_out = self.cwt_stats_layers(decoder_inp.mean(1))  # [B, 2]
        mean = ret["f0_mean"] = stats_out[:, 0]
        std = ret["f0_std"] = stats_out[:, 1]
        cwt_spec = cwt_out[:, :, :10]
        if infer:
            std = std * self.hparams["cwt_std_scale"]
            f0 = self.cwt2f0_norm(cwt_spec, mean, std, mel2ph)
            assert cwt_out.shape[-1] == 11
            uv = cwt_out[:, :, -1] > 0
        ret["f0_denorm"] = f0_denorm = denorm_f0(f0, uv, pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)  # start from 0
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed

    def forward_energy(self, decoder_inp, energy, ret, infer=False):
        ret["energy_pred"] = energy_pred = self.energy_predictor(decoder_inp)[:, :, 0]
        energy_embed_inp = energy_pred if infer else energy
        energy_embed_inp = torch.clamp(
            energy_embed_inp * 256 // 4, min=0, max=255
        ).long()
        energy_embed = self.energy_embed(energy_embed_inp)
        return energy_embed

    def cwt2f0_norm(self, cwt_spec, mean, std, mel2ph):
        _, cwt_scales = get_lf0_cwt(np.ones(10))
        f0 = cwt2f0(cwt_spec, mean, std, cwt_scales)
        f0 = torch.cat([f0] + [f0[:, -1:]] * (mel2ph.shape[1] - f0.shape[1]), 1)
        f0_norm = norm_f0(f0, None)
        return f0_norm

    def ph_states_to_word_states(self, x, ph2word):
        """
        x: [B, T_ph, H]
        ph2word: [B, T_ph] with 0 for padding, 1..W for word index
        return: word_states [B, W, H], word_padding [B, W] (True if pad)
        """
        B, T, H = x.shape
        device = x.device

        W = int(ph2word.max().item()) if ph2word is not None else 0
        if W == 0:
            # fallback: no words
            return x.new_zeros(B, 1, H), torch.ones(B, 1, dtype=torch.bool, device=device)

        # indices: 0..W-1
        idx = (ph2word.clamp_min(0) - 1).clamp_min(0)  # [B, T]
        valid = (ph2word > 0).float()  # [B, T]

        # scatter add
        word_sum = x.new_zeros(B, W, H)
        word_cnt = x.new_zeros(B, W, 1)

        word_sum.scatter_add_(1, idx[:, :, None].expand(-1, -1, H), x * valid[:, :, None])
        word_cnt.scatter_add_(1, idx[:, :, None].expand(-1, -1, 1), valid[:, :, None])

        word_states = word_sum / word_cnt.clamp_min(1.0)
        word_padding = (word_cnt.squeeze(-1) == 0)  # [B, W]
        return word_states, word_padding
