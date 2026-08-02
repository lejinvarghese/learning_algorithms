import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset, load_dataset
from torch.utils.data import Dataset


def _encode(caption: str, seq_len: int) -> torch.Tensor:
    raw = caption.encode("utf-8")[:seq_len]
    ids = torch.zeros(seq_len, dtype=torch.long)
    ids[: len(raw)] = torch.tensor(list(raw), dtype=torch.long)
    return ids


class HFTextDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        n_samples: int = 32,
        seq_len: int = 32,
        frame_size: int = 112,
        num_frames: int = 8,
    ):
        split_name = {"train": "train", "val": "valid", "test": "test"}[split]
        ds = load_dataset("lv12/MultiModalDataset", "fineweb", split=split_name)
        assert isinstance(ds, HFDataset), f"Expected Dataset, got {type(ds)}"
        self.ds: HFDataset = ds.select(range(min(n_samples, len(ds))))
        self.seq_len, self.frame_size, self.num_frames = seq_len, frame_size, num_frames

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        ids = _encode(self.ds[idx]["text"], self.seq_len)
        frames = torch.zeros(self.num_frames, 3, self.frame_size, self.frame_size)
        return ids, frames, torch.tensor(0.0)


class HFImageCaptionDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        n_samples: int = 32,
        seq_len: int = 32,
        frame_size: int = 112,
        num_frames: int = 8,
    ):
        split_name = {"train": "train", "val": "valid", "test": "test"}[split]
        ds = load_dataset("lv12/MultiModalDataset", "coco", split=split_name)
        assert isinstance(ds, HFDataset), f"Expected Dataset, got {type(ds)}"
        self.ds: HFDataset = ds.select(range(min(n_samples, len(ds))))
        self.seq_len, self.frame_size, self.num_frames = seq_len, frame_size, num_frames

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        row = self.ds[idx]
        ids = _encode(row["text"], self.seq_len)

        img = torch.from_numpy(np.array(row["image"].convert("RGB"))).permute(2, 0, 1).float() / 255.0
        img = F.interpolate(img.unsqueeze(0), size=(self.frame_size, self.frame_size), mode="bilinear")
        frames = img.repeat(self.num_frames, 1, 1, 1)
        return ids, frames, torch.tensor(1.0)


class HFVideoCaptionDataset(Dataset):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
