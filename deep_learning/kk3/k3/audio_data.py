"""Audio dataset loader for K3 training."""
import io
import contextlib
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from datasets import load_dataset


def _get_tokenizer():
    """Lazy tokenizer initialization for multiprocessing."""
    global _tokenizer
    if "_tokenizer" not in globals() or _tokenizer is None:
        from k3.tokenizer import get_k3_tokenizer
        _tokenizer = get_k3_tokenizer()
    return _tokenizer


def audio_to_mel_spectrogram(
    waveform: np.ndarray | torch.Tensor | list,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    target_length: int = 3000,
) -> torch.Tensor:
    """
    Convert waveform to mel-spectrogram (Whisper preprocessing).

    Args:
        waveform: Audio waveform (samples,) or (1, samples) or list
        sample_rate: Sample rate (default 16kHz for Whisper)
        n_fft: FFT size
        hop_length: ~10ms hop (16000/160 = 100 fps)
        n_mels: 80 mel bins (Whisper standard)
        target_length: Max frames after conversion

    Returns:
        (n_mels, T) mel-spectrogram, padded/truncated to target_length
    """
    # Convert list to numpy array first
    if isinstance(waveform, list):
        waveform = np.array(waveform, dtype=np.float32)

    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()

    # Ensure 1D or 2D tensor
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() > 2:
        waveform = waveform.squeeze()

    # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # Compute mel-spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    mel = mel_transform(waveform)  # (1, n_mels, T)
    mel = mel.squeeze(0)  # (n_mels, T)

    # Log scale
    mel = torch.log(mel.clamp(min=1e-10))

    # Pad or truncate to target length
    T = mel.shape[1]
    if T < target_length:
        mel = torch.nn.functional.pad(mel, (0, target_length - T))
    else:
        mel = mel[:, :target_length]

    return mel  # (n_mels, target_length)


class HFAudioDataset(Dataset):
    """
    HuggingFace audio dataset loader for K3.

    Dataset: lv12/MultiModalDataset (audioset config)
    Expected format:
    - 'audio': {'array': waveform, 'sampling_rate': int} or raw array
    - 'text': caption/description
    """

    def __init__(self, split: str, max_samples: int, seq_len: int, max_audio_len: int = 3000,
                 num_frames: int = 4, frame_size: int = 112):
        """
        Args:
            split: 'train' or 'validation'
            max_samples: Max samples to load
            seq_len: Max text sequence length
            max_audio_len: Max mel-spec frames (~30s at 100fps → 50fps after conv)
        """
        self.seq_len = seq_len
        self.max_audio_len = max_audio_len

        # Map split names
        hf_split = "train" if split == "train" else "test"

        # Load dataset WITHOUT streaming (enables random access + doesn't load all into memory)
        # Use select() to limit samples instead of loading all then taking
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ds = load_dataset(
                "lv12/MultiModalDataset",
                "audioset",
                split=hf_split,
                streaming=False,  # Non-streaming for random access
            )
            # Only select the samples we need (doesn't load data until __getitem__)
            actual_samples = min(max_samples, len(ds))
            self.data = ds.select(range(actual_samples))

        # Cache dummy tensors to avoid recreating on every __getitem__
        # IMPORTANT: Must match vision dataset shape for batch collation
        self._dummy_frames = torch.zeros(num_frames, 3, frame_size, frame_size)
        self._has_visual = torch.tensor(0.0)
        self._has_audio = torch.tensor(1.0)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Returns (matching vision dataset format):
            ids: (seq_len,) text tokens
            images: dummy tensor (empty, for compatibility)
            has_visual: 0.0 (no images)
            mel: (n_mels, max_audio_len) mel-spectrogram
            has_audio: 1.0 (audio present)
        """
        sample = self.data[idx]

        # Get text caption
        text = sample.get("text", "")
        tokenizer = _get_tokenizer()

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            tokens = tokenizer.encode(text, add_special_tokens=False)[: self.seq_len]

        ids = torch.zeros(self.seq_len, dtype=torch.long)
        ids[: len(tokens)] = torch.tensor(tokens, dtype=torch.long)

        # Get audio
        audio_data = sample["audio"]

        # Handle different audio formats
        try:
            if isinstance(audio_data, dict):
                # HF Audio feature: {'array': np.array, 'sampling_rate': int}
                waveform = audio_data.get("array", [])
                sample_rate = audio_data.get("sampling_rate", 16000)
            elif isinstance(audio_data, (list, np.ndarray)):
                # Raw array, assume 16kHz
                waveform = audio_data
                sample_rate = 16000
            else:
                # Fallback: empty audio
                waveform = []
                sample_rate = 16000

            # Check if waveform is empty or invalid
            if isinstance(waveform, (list, np.ndarray)) and len(waveform) == 0:
                # Empty audio - use silence
                mel = torch.zeros(80, self.max_audio_len)
            else:
                # Convert to mel-spectrogram
                mel = audio_to_mel_spectrogram(
                    waveform,
                    sample_rate=sample_rate,
                    n_mels=80,
                    target_length=self.max_audio_len,
                )
        except Exception as e:
            # If audio conversion fails, use silence
            print(f"Warning: Audio conversion failed for sample {idx}: {e}. Using silence.")
            mel = torch.zeros(80, self.max_audio_len)

        # Return format: (ids, images_dummy, has_visual, audio_mel, has_audio)
        # This matches vision dataset + audio fields
        return ids, self._dummy_frames, self._has_visual, mel, self._has_audio


class MixedModalDataset(Dataset):
    """
    Mixed text/image/audio dataset combining all modalities.
    Randomly samples from text, image, and audio datasets.
    """

    def __init__(self, text_ds, image_ds, audio_ds):
        self.datasets = [text_ds, image_ds, audio_ds]
        self.lengths = [len(ds) for ds in self.datasets]
        self.total = sum(self.lengths)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        # Determine which dataset this index belongs to
        for i, length in enumerate(self.lengths):
            if idx < length:
                sample = self.datasets[i][idx]
                # Pad output to (ids, visual, has_visual, audio, has_audio)
                if len(sample) == 3 and i == 0:  # text dataset
                    ids, dummy1, dummy2 = sample
                    return ids, torch.zeros(1, 1, 1), torch.tensor(0.0), torch.zeros(80, 3000), torch.tensor(0.0)
                elif len(sample) == 3 and i == 1:  # image dataset
                    ids, images, has_visual = sample
                    return ids, images, has_visual, torch.zeros(80, 3000), torch.tensor(0.0)
                else:  # audio dataset
                    ids, mel, has_audio = sample
                    return ids, torch.zeros(1, 1, 1), torch.tensor(0.0), mel, has_audio
            idx -= length
        raise IndexError(f"Index {idx} out of range")
