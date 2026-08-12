"""
Audio encoder for K3 - follows Kimi-Audio design using Whisper encoder.

Architecture:
- Input: raw waveform or mel-spectrogram
- Encoder: Whisper-style conv + transformer
- Output: continuous acoustic features projected to K3 hidden_dim
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from k3.layers import RMSNorm


class WhisperStyleConv(nn.Module):
    """Whisper's conv stem: 2 conv layers that downsample to ~50Hz."""
    def __init__(self, n_mels: int = 80, hidden: int = 256):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_mels, T) mel-spectrogram
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        return x.transpose(1, 2)  # (B, T/2, hidden)


class AudioTransformerBlock(nn.Module):
    """Simplified transformer block for audio encoding."""
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
        # Self-attention with residual
        x = x + self.attn(self.attn_norm(x), self.attn_norm(x), self.attn_norm(x))[0]
        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))
        return x


class K3AudioEncoder(nn.Module):
    """
    Audio encoder for K3, following Kimi-Audio's Whisper-based design.

    Input: mel-spectrogram (B, n_mels, T)
    Output: (B, audio_seq_len, hidden_dim) - continuous acoustic features
    """
    def __init__(self, cfg):
        super().__init__()
        self.n_mels = cfg.audio_n_mels
        self.audio_hidden = cfg.audio_hidden

        # Whisper-style conv stem (downsamples ~2x)
        self.conv = WhisperStyleConv(self.n_mels, self.audio_hidden)

        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, cfg.audio_max_frames, self.audio_hidden) * 0.02
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            AudioTransformerBlock(self.audio_hidden, cfg.audio_heads)
            for _ in range(cfg.audio_layers)
        ])

        self.final_norm = RMSNorm(self.audio_hidden)

        # Project to K3 hidden_dim
        self.projector = nn.Sequential(
            nn.Linear(self.audio_hidden, self.audio_hidden * 2),
            nn.GELU(),
            nn.Linear(self.audio_hidden * 2, cfg.hidden_dim),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, n_mels, T) mel-spectrogram

        Returns:
            (B, audio_seq_len, hidden_dim) acoustic features
        """
        B = mel.shape[0]

        # Conv stem
        x = self.conv(mel)  # (B, T/2, audio_hidden)
        T = x.shape[1]

        # Add positional encoding
        x = x + self.pos_embed[:, :T]

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        # Project to K3 hidden_dim
        return self.projector(x)  # (B, T/2, hidden_dim)


def mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
) -> torch.Tensor:
    """
    Convert waveform to mel-spectrogram (Whisper preprocessing).

    Args:
        waveform: (B, num_samples) or (num_samples,)
        sample_rate: 16kHz (Whisper standard)
        n_fft: FFT size
        hop_length: ~10ms hop (16000/160 = 100 fps → 50fps after conv)
        n_mels: 80 mel bins (Whisper standard)

    Returns:
        (B, n_mels, T) mel-spectrogram
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Compute spectrogram
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=torch.hann_window(n_fft, device=waveform.device),
        return_complex=True,
    )

    # Power spectrogram
    power = spec.abs().pow(2)

    # Mel filterbank (simplified - in practice, use torchaudio.transforms.MelScale)
    # For now, just return power spec as placeholder
    # TODO: implement proper mel filterbank or use torchaudio
    mel = power[..., :n_mels]  # (B, n_mels, T)

    # Log mel-spectrogram
    mel = torch.log(mel.clamp(min=1e-10))

    return mel
