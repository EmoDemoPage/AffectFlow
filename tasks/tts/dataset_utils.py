import os
import torch.optim
import torch.utils.data
import numpy as np
import torch
import torch.optim
import torch.utils.data
import torch.distributions
from utils.audio.pitch.utils import norm_interp_f0, denorm_f0
from utils.commons.dataset_utils import (
    BaseDataset,
    collate_1d_or_2d,
    collate_1d,
    collate_2d,
)
from utils.commons.indexed_datasets import IndexedDataset
from utils.text.text_encoder import build_token_encoder
from utils.text import intersperse


class BaseSpeechDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(shuffle)
        from utils.commons.hparams import hparams

        self.data_dir = hparams["binary_data_dir"] if data_dir is None else data_dir
        self.prefix = prefix
        self.hparams = hparams
        self.indexed_ds = None
        if items is not None:
            self.indexed_ds = items
            self.sizes = [1] * len(items)
            self.avail_idxs = list(range(len(self.sizes)))
        else:
            self.sizes = np.load(f"{self.data_dir}/{self.prefix}_lengths.npy")
            if prefix == "test" and len(hparams["test_ids"]) > 0:
                self.avail_idxs = hparams["test_ids"]
                # self.avail_idxs = list(range(0, 100)) 
            else:
                self.avail_idxs = list(range(len(self.sizes)))
            if prefix == "train" and hparams["min_frames"] > 0:
                self.avail_idxs = [
                    x for x in self.avail_idxs if self.sizes[x] >= hparams["min_frames"]
                ]
            self.sizes = [self.sizes[i] for i in self.avail_idxs]

    def _get_item(self, index):
        if hasattr(self, "avail_idxs") and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f"{self.data_dir}/{self.prefix}")
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        assert len(item["mel"]) == self.sizes[index], (
            len(item["mel"]),
            self.sizes[index],
        )
        max_frames = hparams["max_frames"]
        spec = torch.Tensor(item["mel"])[:max_frames]
        max_frames = (
            spec.shape[0] // hparams["frames_multiple"] * hparams["frames_multiple"]
        )
        spec = spec[:max_frames]
        ph_token = torch.LongTensor(item["ph_token"][: hparams["max_input_tokens"]])
        sample = {
            "id": index,
            "item_name": item["item_name"],
            "text": item["txt"],
            "txt_token": ph_token,
            "mel": spec,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
        }
        if hparams["use_spk_embed"]:
            sample["spk_embed"] = torch.Tensor(item["spk_embed"])
        if hparams["use_spk_id"]:
            sample["spk_id"] = int(item["spk_id"])
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        id = torch.LongTensor([s["id"] for s in samples])
        item_names = [s["item_name"] for s in samples]
        text = [s["text"] for s in samples]
        txt_tokens = collate_1d_or_2d([s["txt_token"] for s in samples], 0)
        mels = collate_1d_or_2d([s["mel"] for s in samples], 0.0)
        txt_lengths = torch.LongTensor([s["txt_token"].numel() for s in samples])
        mel_lengths = torch.LongTensor([s["mel"].shape[0] for s in samples])

        batch = {
            "id": id,
            "item_name": item_names,
            "nsamples": len(samples),
            "text": text,
            "txt_tokens": txt_tokens,
            "txt_lengths": txt_lengths,
            "mels": mels,
            "mel_lengths": mel_lengths,
        }

        if hparams["use_spk_embed"]:
            spk_embed = torch.stack([s["spk_embed"] for s in samples])
            batch["spk_embed"] = spk_embed
        if hparams["use_spk_id"]:
            spk_ids = torch.LongTensor([s["spk_id"] for s in samples])
            batch["spk_ids"] = spk_ids
        return batch


class FastSpeechDataset(BaseSpeechDataset):
    def __getitem__(self, index):
        sample = super(FastSpeechDataset, self).__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams
        mel = sample["mel"]
        T = mel.shape[0]
        ph_token = sample["txt_token"]
        sample["mel2ph"] = mel2ph = torch.LongTensor(item["mel2ph"])[:T]
        if hparams["use_pitch_embed"]:
            assert "f0" in item
            pitch = torch.LongTensor(item.get(hparams.get("pitch_key", "pitch")))[:T]
            f0, uv = norm_interp_f0(item["f0"][:T])
            uv = torch.FloatTensor(uv)
            f0 = torch.FloatTensor(f0)
            if hparams["pitch_type"] == "ph":
                if "f0_ph" in item:
                    f0 = torch.FloatTensor(item["f0_ph"])
                else:
                    f0 = denorm_f0(f0, None)
                f0_phlevel_sum = (
                    torch.zeros_like(ph_token).float().scatter_add(0, mel2ph - 1, f0)
                )
                f0_phlevel_num = (
                    torch.zeros_like(ph_token)
                    .float()
                    .scatter_add(0, mel2ph - 1, torch.ones_like(f0))
                    .clamp_min(1)
                )
                f0_ph = f0_phlevel_sum / f0_phlevel_num
                f0, uv = norm_interp_f0(f0_ph)
        else:
            f0, uv, pitch = None, None, None
        sample["f0"], sample["uv"], sample["pitch"] = f0, uv, pitch
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(FastSpeechDataset, self).collater(samples)
        hparams = self.hparams
        if hparams["use_pitch_embed"]:
            f0 = collate_1d_or_2d([s["f0"] for s in samples], 0.0)
            pitch = collate_1d_or_2d([s["pitch"] for s in samples])
            uv = collate_1d_or_2d([s["uv"] for s in samples])
        else:
            f0, uv, pitch = None, None, None
        mel2ph = collate_1d_or_2d([s["mel2ph"] for s in samples], 0.0)
        batch.update(
            {
                "mel2ph": mel2ph,
                "pitch": pitch,
                "f0": f0,
                "uv": uv,
            }
        )
        return batch


class GradTTSDataset(FastSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(prefix, shuffle, items, data_dir)
        data_dir = self.hparams["processed_data_dir"]
        self.token_encoder = build_token_encoder(f"{data_dir}/phone_set.json")

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        ph_token = sample["txt_token"]
        ph_token = intersperse(ph_token, len(self.token_encoder))
        ph_token = torch.IntTensor(ph_token)
        sample["txt_token"] = ph_token
        return sample

    def collater(self, samples):
        batch = super().collater(samples)
        return batch


class FastSpeech2Dataset(FastSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(prefix, shuffle, items, data_dir)
        self.pitch_type = self.hparams.get("pitch_type")

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams
        mel = sample["mel"]
        T = mel.shape[0]
        sample["energy"] = (mel.exp() ** 2).sum(-1).sqrt()
        cwt_spec = torch.Tensor(item["cwt_spec"])[:T]
        f0_mean = item.get("f0_mean", item.get("cwt_mean"))
        f0_std = item.get("f0_std", item.get("cwt_std"))
        sample.update({"cwt_spec": cwt_spec, "f0_mean": f0_mean, "f0_std": f0_std})
        return sample

    def collater(self, samples):
        batch = super().collater(samples)
        energy = collate_1d([s["energy"] for s in samples], 0.0)
        batch.update({"energy": energy})
        cwt_spec = collate_2d([s["cwt_spec"] for s in samples])
        f0_mean = torch.Tensor([s["f0_mean"] for s in samples])
        f0_std = torch.Tensor([s["f0_std"] for s in samples])
        batch.update({"cwt_spec": cwt_spec, "f0_mean": f0_mean, "f0_std": f0_std})
        return batch

class AffectFlow2Dataset(FastSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(prefix, shuffle, items, data_dir)
        self.pitch_type = self.hparams.get("pitch_type")

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams
        mel = sample["mel"]
        T = mel.shape[0]
        sample["mel2word"] = mel2word = torch.LongTensor(item["mel2word"])[:T]
        sample["ph2word"] = ph2word = torch.LongTensor(item["ph2word"])

        selected_pt_path = item['item_name'] + ".pt"
        # VAD Extract
        # wordsum_VAD_pad
        wordsum_VAD_pad_path = "/dataset/dailytalk_VAD_prml/wordsum_VAD_pad"
        wordsum_VAD_pad_pt_full_path = os.path.join(wordsum_VAD_pad_path, selected_pt_path)
        sample["wordsum_VAD_pad"] = torch.load(wordsum_VAD_pad_pt_full_path).detach() # word_VAD_pad: (U_len, 3)

        # wordsum_VAD_mask
        wordsum_VAD_mask_path = "/dataset/dailytalk_VAD_prml/wordsum_VAD_mask"
        wordsum_VAD_mask_pt_full_path = os.path.join(wordsum_VAD_mask_path, selected_pt_path)
        sample["wordsum_VAD_mask"] = torch.load(wordsum_VAD_mask_pt_full_path).detach() # word_VAD_pad: (U_len)
        
        # word
        word_VAD_path = "/dataset/dailytalk_VAD_prml/word2utterance"
        word_pt_full_path = os.path.join(word_VAD_path, selected_pt_path)
        word_VAD = torch.load(word_pt_full_path).detach()
        
        gt_u = torch.sort(torch.unique(mel2word[mel2word > 0])).values  # (W_len)
        Wmax = int(ph2word.max().item())

        dense = word_VAD.new_zeros(Wmax + 1, 3) 
        dense[gt_u] = word_VAD  
        sample["word_VAD_pad"] = word_VAD_pad = dense[1:]  # word_VAD_pad: (W_len, 3)

        mask = (word_VAD_pad.abs().sum(dim=-1) > 0).long()  # (W_len)
        sample["word_VAD_mask"] = mask # word_VAD_mask: (W_len)

        # Bert Extract
        if hparams["bert_model"] == "bert":
            if hparams["token_pad"]:
                # wordsum
                wordsum_bert_path = "/dataset/dailytalk_bert/long_dialogue_pad"
                wordsum_bert_pt_full_path = os.path.join(wordsum_bert_path, selected_pt_path)
                sample["wordsum_bert_emb"] = torch.load(wordsum_bert_pt_full_path, map_location="cpu").detach() # wordsum_bert_emb: (U_len, 768)
                
                # qmask
                qmask_bert_path = "/dataset/dailytalk_bert/long_qmask"
                qmask_bert_pt_full_path = os.path.join(qmask_bert_path, selected_pt_path)
                sample["qmask_bert_emb"] = torch.load(qmask_bert_pt_full_path, map_location="cpu").detach() # qmask_bert_emb: (U_len, 2)
                
            else:
                # wordsum
                wordsum_bert_path = "/dataset/dailytalk_bert/long_dialogue_nopad"
                wordsum_bert_pt_full_path = os.path.join(wordsum_bert_path, selected_pt_path)
                sample["wordsum_bert_emb"] = torch.load(wordsum_bert_pt_full_path, map_location="cpu").detach() # wordsum_bert_emb: (U_len, 768)
                
                # qmask
                qmask_bert_path = "/dataset/dailytalk_bert/long_qmask"
                qmask_bert_pt_full_path = os.path.join(qmask_bert_path, selected_pt_path)
                sample["qmask_bert_emb"] = torch.load(qmask_bert_pt_full_path, map_location="cpu").detach() # qmask_bert_emb: (U_len, 2)
                
        elif hparams["bert_model"] == "roberta":
            if hparams["token_pad"]:
                # wordsum
                wordsum_bert_path = "/dataset/dailytalk_roberta/long_dialogue_pad"
                wordsum_bert_pt_full_path = os.path.join(wordsum_bert_path, selected_pt_path)
                sample["wordsum_bert_emb"] = torch.load(wordsum_bert_pt_full_path, map_location="cpu").detach() # wordsum_bert_emb: (U_len, 768)
                
                # qmask
                qmask_bert_path = "/dataset/dailytalk_roberta/long_qmask"
                qmask_bert_pt_full_path = os.path.join(qmask_bert_path, selected_pt_path)
                sample["qmask_bert_emb"] = torch.load(qmask_bert_pt_full_path, map_location="cpu").detach() # qmask_bert_emb: (U_len, 2)
            
            else:
                # wordsum
                wordsum_bert_path = "/dataset/dailytalk_roberta/long_dialogue_nopad"
                wordsum_bert_pt_full_path = os.path.join(wordsum_bert_path, selected_pt_path)
                sample["wordsum_bert_emb"] = torch.load(wordsum_bert_pt_full_path, map_location="cpu").detach() # wordsum_bert_emb: (U_len, 768)
                
                # qmask
                qmask_bert_path = "/dataset/dailytalk_roberta/long_qmask"
                qmask_bert_pt_full_path = os.path.join(qmask_bert_path, selected_pt_path)
                sample["qmask_bert_emb"] = torch.load(qmask_bert_pt_full_path, map_location="cpu").detach() # qmask_bert_emb: (U_len, 2)

        sample["Wsum_len"] = sample["wordsum_bert_emb"].size(0)
        sample["energy"] = (mel.exp() ** 2).sum(-1).sqrt()
        cwt_spec = torch.Tensor(item["cwt_spec"])[:T]
        f0_mean = item.get("f0_mean", item.get("cwt_mean"))
        f0_std = item.get("f0_std", item.get("cwt_std"))
        sample.update({"cwt_spec": cwt_spec, "f0_mean": f0_mean, "f0_std": f0_std})
        return sample

    def collater(self, samples):
        batch = super().collater(samples)
        energy = collate_1d([s["energy"] for s in samples], 0.0)
        batch.update({"energy": energy})
        cwt_spec = collate_2d([s["cwt_spec"] for s in samples])
        f0_mean = torch.Tensor([s["f0_mean"] for s in samples])
        f0_std = torch.Tensor([s["f0_std"] for s in samples])
        mel2word = collate_1d_or_2d([s["mel2word"] for s in samples], 0.0)
        ph2word = collate_1d_or_2d([s["ph2word"] for s in samples], 0.0)

        # VAD        
        word_VAD_pad = collate_1d_or_2d([s['word_VAD_pad'] for s in samples], 0.0)
        word_VAD_mask = collate_1d_or_2d([s['word_VAD_mask'] for s in samples], 0.0)
        wordsum_VAD_pad = collate_1d_or_2d([s['wordsum_VAD_pad'] for s in samples], 0.0)
        wordsum_VAD_mask = collate_1d_or_2d([s['wordsum_VAD_mask'] for s in samples], 0.0)
        
        # bert
        wordsum_bert_emb = collate_1d_or_2d([s['wordsum_bert_emb'] for s in samples], 0.0)
        qmask_bert_emb = collate_1d_or_2d([s['qmask_bert_emb'] for s in samples], 0.0)
        
        Wsum_len = torch.LongTensor([s["Wsum_len"] for s in samples])
        Wsum_max = wordsum_bert_emb.size(1)
        wordsum_mask = (torch.arange(Wsum_max)[None, :] < Wsum_len[:, None]).long()
        
        batch.update({"cwt_spec": cwt_spec, "f0_mean": f0_mean, "f0_std": f0_std, "mel2word": mel2word, "wordsum_VAD_pad": wordsum_VAD_pad, "wordsum_VAD_mask": wordsum_VAD_mask, "word_VAD_pad": word_VAD_pad, "word_VAD_mask": word_VAD_mask, "ph2word": ph2word, "word_VAD_mask": word_VAD_mask, "wordsum_bert_emb": wordsum_bert_emb, "qmask_bert_emb": qmask_bert_emb, "wordsum_mask": wordsum_mask})
        return batch