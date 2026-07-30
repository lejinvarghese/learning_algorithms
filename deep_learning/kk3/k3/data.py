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
    def __init__(
        self, n_samples: int = 32, seq_len: int = 32, image_size: int = 112,
        num_frames: int = 8, seed: int = 0,
    ):
        rng = random.Random(seed)
        self.seq_len, self.image_size, self.num_frames = seq_len, image_size, num_frames
        self.samples = [(rng.choice(list(COLORS)), rng.choice(POSITIONS)) for _ in range(n_samples)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        color, position = self.samples[idx]
        caption = f"a {color} square in the {position}"
        raw = caption.encode("utf-8")[: self.seq_len]
        ids = torch.zeros(self.seq_len, dtype=torch.long)
        ids[: len(raw)] = torch.tensor(list(raw), dtype=torch.long)
        frames = render_image(color, position, self.image_size).unsqueeze(0).repeat(self.num_frames, 1, 1, 1)
        return ids, frames, torch.tensor(1.0)
