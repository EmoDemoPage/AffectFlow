import torch
import torch.nn.functional as F


def build_word_mask(x2word, y2word):
    return (x2word[:, :, None] == y2word[:, None, :]).long()


def mel2ph_to_mel2word(mel2ph, ph2word):
    mel2word = (ph2word - 1).gather(1, (mel2ph - 1).clamp(min=0)) + 1
    mel2word = mel2word * (mel2ph > 0).long()
    return mel2word


def clip_mel2token_to_multiple(mel2token, frames_multiple):
    max_frames = mel2token.shape[1] // frames_multiple * frames_multiple
    mel2token = mel2token[:, :max_frames]
    return mel2token


def expand_states(h, mel2token):
    h = F.pad(h, [0, 0, 1, 0])
    mel2token_ = mel2token[..., None].repeat([1, 1, h.shape[-1]])
    h = torch.gather(h, 1, mel2token_)  # [B, T, H]
    return h

def word_vad_to_frame_vad(word_VAD, pred_mel2word, gt_mel2word):
    """
    word_VAD:     (B, Nw, 3)   
    pred_mel2word:(B, T)       
    gt_mel2word:  (B, T)       
    return: frame_VAD (B, T, 3)
    """
    B, T = pred_mel2word.shape
    D = word_VAD.shape[-1]
    frame = word_VAD.new_zeros(B, T, D)

    for b in range(B):
        gt_ids = gt_mel2word[b]
        gt_valid = gt_ids > 0
        if gt_valid.sum() == 0:
            continue

        # GT에서 실제 등장한 word id들 (정렬). 예: [1,2,4,5]
        gt_u = torch.sort(torch.unique(gt_ids[gt_valid])).values
        nw_eff = gt_u.numel()

        # word_VAD row 수가 gt_u 개수와 맞아야 함
        if word_VAD.shape[1] < nw_eff:
            raise ValueError(f"[word_vad_to_frame_vad_safe] word_VAD rows 부족: "
                             f"need={nw_eff}, have={word_VAD.shape[1]} (batch={b})")

        # pred ids
        pred_ids = pred_mel2word[b]
        pred_valid = pred_ids > 0

        # pred_ids가 gt_u 범위 안일 때만 searchsorted 의미 있음
        umin = int(gt_u.min().item())
        umax = int(gt_u.max().item())
        in_range = pred_valid & (pred_ids >= umin) & (pred_ids <= umax)

        # 일단 범위 내는 idx 계산 (0..nw_eff-1)
        idx = torch.searchsorted(gt_u, pred_ids.clamp(min=umin, max=umax))

        # 하지만 pred_ids 값이 gt_u에 "정확히 존재"하는지 확인해야 함
        # (예: gt_u=[1,2,4,5], pred_id=3이면 searchsorted=2지만 존재하지 않음)
        exists = in_range & (gt_u[idx] == pred_ids)

        # 존재하는 프레임만 채움, 나머지는 0 유지
        frame[b, exists] = word_VAD[b, idx[exists]]

    return frame

def word_vad_to_frame_vad_simple(word_VAD, mel2word):
    """
    word_VAD: [B, W, 3]    (row=word_id-1)
    mel2word: [B, T]       (0 pad, 1..W)
    """
    B, T = mel2word.shape
    D = word_VAD.shape[-1]
    out = word_VAD.new_zeros(B, T, D)

    valid = mel2word > 0
    idx0 = (mel2word[valid] - 1).long()  # 0..W-1
    out[valid] = word_VAD[torch.arange(B, device=word_VAD.device).repeat_interleave(valid.sum(1)),
                          idx0]
    return out
