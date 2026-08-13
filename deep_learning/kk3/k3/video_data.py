"""Video dataset loader for K3 training."""

import contextlib
import io
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


def _get_tokenizer():
    """Lazy tokenizer initialization for multiprocessing."""
    global _tokenizer
    if "_tokenizer" not in globals() or _tokenizer is None:
        from k3.tokenizer import get_k3_tokenizer
        _tokenizer = get_k3_tokenizer()
    return _tokenizer


class HFVideoDataset(Dataset):
    """
    HuggingFace video dataset loader for K3.

    Dataset: lv12/MultiModalDataset (openvid config)
    Format: 4 frames per video @ 112x112 RGB stored as Image features
    """

    def __init__(self, split: str, max_samples: int, seq_len: int,
                 num_frames: int = 4, frame_size: int = 112):
        """
        Args:
            split: 'train' or 'val'
            max_samples: Max samples to load
            seq_len: Max text sequence length
            num_frames: Number of frames per video (must match preprocessing)
            frame_size: Frame resolution (must match preprocessing)
        """
        self.seq_len = seq_len
        self.num_frames = num_frames
        self.frame_size = frame_size

        # Map split names
        hf_split = "train" if split == "train" else "validation"

        # Load dataset
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ds = load_dataset(
                "lv12/MultiModalDataset",
                "openvid",
                split=hf_split,
            )
            actual_samples = min(max_samples, len(ds))
            self.data = ds.select(range(actual_samples))

        # Cache dummy audio tensors (no audio in video dataset)
        self._dummy_audio = torch.zeros(80, 1500)
        self._has_audio = torch.tensor(0.0)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Returns (matching other dataset formats):
            ids: (seq_len,) text tokens
            frames: (num_frames, 3, frame_size, frame_size) video frames
            has_visual: 1.0 (video present)
            audio_mel: dummy tensor (no audio)
            has_audio: 0.0 (no audio)
        """
        sample = self.data[idx]

        # Get text caption
        caption = sample.get("caption", "")
        tokenizer = _get_tokenizer()

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            tokens = tokenizer.encode(caption, add_special_tokens=False)[: self.seq_len]

        ids = torch.zeros(self.seq_len, dtype=torch.long)
        ids[: len(tokens)] = torch.tensor(tokens, dtype=torch.long)

        # Load and stack frames
        frames = []
        for i in range(self.num_frames):
            # HF Image feature automatically loads as PIL Image
            img = sample[f'frame_{i}'].convert('RGB')
            img_array = np.array(img)

            # Convert to tensor (H, W, 3) -> (3, H, W)
            frame_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            frames.append(frame_tensor)

        frames = torch.stack(frames)  # (num_frames, 3, H, W)

        # Return format: (ids, frames, has_visual, audio_mel, has_audio)
        return ids, frames, torch.tensor(1.0), self._dummy_audio, self._has_audio
