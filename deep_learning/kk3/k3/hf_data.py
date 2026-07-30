import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset


def _encode(caption: str, seq_len: int) -> torch.Tensor:
    raw = caption.encode("utf-8")[:seq_len]
    ids = torch.zeros(seq_len, dtype=torch.long)
    ids[: len(raw)] = torch.tensor(list(raw), dtype=torch.long)
    return ids


class HFTextDataset(Dataset):
    def __init__(
        self, split: str = "train", n_samples: int = 32, seq_len: int = 32,
        frame_size: int = 112, num_frames: int = 8,
    ):
        split = {"train": "train", "val": "validation", "test": "test"}[split]
        stream = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split, streaming=True)
        stream = stream.filter(lambda r: len(r["text"].strip()) > 20)
        self.samples = [r["text"] for r in stream.take(n_samples)]
        self.seq_len, self.frame_size, self.num_frames = seq_len, frame_size, num_frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ids = _encode(self.samples[idx], self.seq_len)
        frames = torch.zeros(self.num_frames, 3, self.frame_size, self.frame_size)
        return ids, frames, torch.tensor(0.0)


class HFImageCaptionDataset(Dataset):
    _TRAIN_POOL = 512  # svjack/pokemon-blip-captions-en-zh has ~833 rows in its one "train" split

    def __init__(
        self, split: str = "train", n_samples: int = 32, seq_len: int = 32,
        frame_size: int = 112, num_frames: int = 8,
    ):
        stream = load_dataset("svjack/pokemon-blip-captions-en-zh", split="train", streaming=True)
        offset = 0 if split == "train" else self._TRAIN_POOL
        self.samples = list(stream.skip(offset).take(n_samples))
        self.seq_len, self.frame_size, self.num_frames = seq_len, frame_size, num_frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        ids = _encode(row["en_text"], self.seq_len)

        img = torch.from_numpy(np.array(row["image"].convert("RGB"))).permute(2, 0, 1).float() / 255.0
        img = F.interpolate(img.unsqueeze(0), size=(self.frame_size, self.frame_size), mode="bilinear")
        frames = img.repeat(self.num_frames, 1, 1, 1)
        return ids, frames, torch.tensor(1.0)


# Not wired up yet. Add a video-caption source here with the same contract as the classes
# above: __getitem__ returns (ids: LongTensor[seq_len], frames: FloatTensor[num_frames, 3,
# frame_size, frame_size] in [0, 1], has_visual: scalar FloatTensor 1.0). See k3/video.py's
# sample_frames() for the decode-and-resize pattern.
class HFVideoCaptionDataset(Dataset):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
