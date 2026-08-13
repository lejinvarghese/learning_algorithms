import torch
import torch.nn as nn
import torch.nn.functional as F
from k3.layers import RMSNorm


class WhisperStyleConv(nn.Module):
    def __init__(self, n_mels: int = 80, hidden: int = 256):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        return x.transpose(1, 2)


class AudioTransformerBlock(nn.Module):
    def __init__(self, hidden: int, num_heads: int):
        super().__init__()
        self.attn_norm = RMSNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.ffn_norm = RMSNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), self.attn_norm(x), self.attn_norm(x))[0]
        x = x + self.ffn(self.ffn_norm(x))
        return x


class K3AudioEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv = WhisperStyleConv(cfg.audio_n_mels, cfg.audio_hidden)
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.audio_max_frames, cfg.audio_hidden) * 0.02)
        self.layers = nn.ModuleList([
            AudioTransformerBlock(cfg.audio_hidden, cfg.audio_heads) for _ in range(cfg.audio_layers)
        ])
        self.final_norm = RMSNorm(cfg.audio_hidden)
        self.projector = nn.Sequential(
            nn.Linear(cfg.audio_hidden, cfg.audio_hidden * 2),
            nn.GELU(),
            nn.Linear(cfg.audio_hidden * 2, cfg.hidden_dim),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.conv(mel)
        B, T, _ = x.shape
        if T <= self.pos_embed.shape[1]:
            x = x + self.pos_embed[:, :T]
        else:
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2), size=T, mode='linear', align_corners=False
            ).transpose(1, 2)
            x = x + pos
        for layer in self.layers:
            x = layer(x)
        return self.projector(self.final_norm(x))


def audio_to_mel_spectrogram(waveform, sample_rate: int = 16000, n_fft: int = 400,
                             hop_length: int = 160, n_mels: int = 80) -> torch.Tensor:
    import numpy as np
    import torchaudio

    if isinstance(waveform, list):
        waveform = np.array(waveform, dtype=np.float32)
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() > 2:
        waveform = waveform.squeeze()

    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )(waveform).squeeze(0)

    return torch.log(mel.clamp(min=1e-10))
