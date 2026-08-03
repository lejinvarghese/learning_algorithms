import contextlib
import io
import logging
import numpy as np
import os
import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset, load_dataset
from torch.utils.data import Dataset
import warnings

# Suppress transformers verbosity
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from .tokenizer import get_k3_tokenizer

# Global tokenizer instance - lazily initialized per process/worker
_tokenizer = None


def _get_tokenizer():
    """Lazy tokenizer initialization for multi-process safety."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = get_k3_tokenizer()
    return _tokenizer


def _encode(caption: str, seq_len: int) -> torch.Tensor:
    """Encode text using K3 tokenizer (suppress verbose output)."""
    tokenizer = _get_tokenizer()
    # Redirect stdout/stderr to suppress tokenizer debug prints
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        tokens = tokenizer.encode(caption, add_special_tokens=False)[:seq_len]
    ids = torch.zeros(seq_len, dtype=torch.long)
    ids[:len(tokens)] = torch.tensor(tokens, dtype=torch.long)
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

        # Limit to available data
        actual_samples = min(n_samples, len(ds))
        if actual_samples < n_samples:
            print(f"⚠ HFTextDataset: requested {n_samples} samples, only {len(ds)} available (using {actual_samples})")

        self.ds: HFDataset = ds.select(range(actual_samples))
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

        # Limit to available data
        actual_samples = min(n_samples, len(ds))
        if actual_samples < n_samples:
            print(f"⚠ HFImageCaptionDataset: requested {n_samples} samples, only {len(ds)} available (using {actual_samples})")

        self.ds: HFDataset = ds.select(range(actual_samples))
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
