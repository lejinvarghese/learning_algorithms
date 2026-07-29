import urllib.request
from pathlib import Path

import av
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

CACHE_DIR = Path.home() / ".cache" / "k3-toy-videos"

CLIPS = [
    ("https://huggingface.co/datasets/hf-internal-testing/tiny-video-dataset/resolve/main/videos/00001.mp4",
     "hiker.mp4", "a hiker standing on a mountain peak at sunrise"),
    ("https://huggingface.co/datasets/nielsr/video-demo/resolve/main/eating_spaghetti.mp4",
     "spaghetti.mp4", "a person eating spaghetti"),
    ("https://huggingface.co/datasets/sayakpaul/ucf101-subset/resolve/main/v_BabyCrawling_g19_c02.avi",
     "baby_crawling.avi", "a baby crawling"),
    ("https://huggingface.co/datasets/sayakpaul/ucf101-subset/resolve/main/v_BasketballDunk_g14_c06.avi",
     "basketball_dunk.avi", "a basketball player dunking"),
]


def sample_frames(path, num_frames: int, size: int) -> torch.Tensor:
    with av.open(str(path)) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(container.streams.video[0])]
    if not frames:
        raise ValueError(f"no decodable frames in {path}")
    idx = torch.linspace(0, len(frames) - 1, num_frames).round().long()
    picked = torch.stack([torch.from_numpy(frames[i]) for i in idx])
    picked = picked.permute(0, 3, 1, 2).float() / 255.0
    return F.interpolate(picked, size=(size, size), mode="bilinear", align_corners=False)


def _ensure_downloaded() -> list:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    clips = []
    for url, name, caption in CLIPS:
        dest = CACHE_DIR / name
        if not dest.exists():
            urllib.request.urlretrieve(url, dest)
        clips.append((dest, caption))
    return clips


class RealVideoDataset(Dataset):
    def __init__(self, num_frames: int = 8, frame_size: int = 112, seq_len: int = 32, split: str = "train"):
        clips = _ensure_downloaded()
        self.samples = clips[:-1] if split == "train" else clips[-1:]
        self.num_frames, self.frame_size, self.seq_len = num_frames, frame_size, seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, caption = self.samples[idx]
        raw = caption.encode("utf-8")[: self.seq_len]
        ids = torch.zeros(self.seq_len, dtype=torch.long)
        ids[: len(raw)] = torch.tensor(list(raw), dtype=torch.long)
        return ids, sample_frames(path, self.num_frames, self.frame_size)
