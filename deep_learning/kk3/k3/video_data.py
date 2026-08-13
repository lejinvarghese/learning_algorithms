import contextlib
import io
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


def _get_tokenizer():
    global _tokenizer
    if "_tokenizer" not in globals() or _tokenizer is None:
        from k3.tokenizer import get_k3_tokenizer
        _tokenizer = get_k3_tokenizer()
    return _tokenizer


class HFVideoDataset(Dataset):
    def __init__(self, split: str, max_samples: int, seq_len: int, num_frames: int = 4, frame_size: int = 112):
        self.seq_len = seq_len
        self.num_frames = num_frames
        self.frame_size = frame_size

        hf_split = "train" if split == "train" else "validation"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ds = load_dataset("lv12/MultiModalDataset", "openvid", split=hf_split)
            self.data = ds.select(range(min(max_samples, len(ds))))

        self._dummy_audio = torch.zeros(80, 1500)
        self._has_audio = torch.tensor(0.0)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        caption = sample.get("caption", "")
        tokenizer = _get_tokenizer()

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            tokens = tokenizer.encode(caption, add_special_tokens=False)[: self.seq_len]

        ids = torch.zeros(self.seq_len, dtype=torch.long)
        ids[: len(tokens)] = torch.tensor(tokens, dtype=torch.long)

        frames = []
        for i in range(self.num_frames):
            img = sample[f'frame_{i}'].convert('RGB')
            img_array = np.array(img)
            frame_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            frames.append(frame_tensor)

        frames = torch.stack(frames)
        return ids, frames, torch.tensor(1.0), self._dummy_audio, self._has_audio
