"""Tiny procedural multimodal dataset for smoke-testing K3-Micro: a colored
block on a canvas, paired with a caption describing it. No downloads, no
extra dependencies (pure torch) -- just enough structure (image content
actually correlates with the caption) to sanity-check that the vision
splice and text loss are both learning something, not just running."""
import random

import torch
from torch.utils.data import Dataset

COLORS = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
}
POSITIONS = ["top-left", "top-right", "bottom-left", "bottom-right", "center"]


def render_image(color: str, position: str, size: int = 112) -> torch.Tensor:
    img = torch.full((3, size, size), 0.1)
    block = size // 3
    if position == "center":
        y0 = x0 = (size - block) // 2
    else:
        ys, xs = position.split("-")
        y0 = 4 if ys == "top" else size - block - 4
        x0 = 4 if xs == "left" else size - block - 4
    for ch, val in enumerate(COLORS[color]):
        img[ch, y0 : y0 + block, x0 : x0 + block] = val
    return img


class ToyMultimodalDataset(Dataset):
    """`n_samples` (caption, image) pairs; captions are byte-encoded so no
    external tokenizer/vocab file is needed (bytes are < 256, comfortably
    inside any K3MicroConfig.vocab_size)."""

    def __init__(self, n_samples: int = 32, seq_len: int = 32, image_size: int = 112, seed: int = 0):
        rng = random.Random(seed)
        self.seq_len, self.image_size = seq_len, image_size
        self.samples = [(rng.choice(list(COLORS)), rng.choice(POSITIONS)) for _ in range(n_samples)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        color, position = self.samples[idx]
        caption = f"a {color} square in the {position}"
        raw = caption.encode("utf-8")[: self.seq_len]
        ids = torch.zeros(self.seq_len, dtype=torch.long)
        ids[: len(raw)] = torch.tensor(list(raw), dtype=torch.long)
        return ids, render_image(color, position, self.image_size)
