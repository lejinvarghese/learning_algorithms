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


def audio_to_mel_spectrogram(waveform, sample_rate: int = 16000, n_fft: int = 400,
                             hop_length: int = 160, n_mels: int = 80, target_length: int = 3000) -> torch.Tensor:
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

    mel = torch.log(mel.clamp(min=1e-10))

    T = mel.shape[1]
    if T < target_length:
        mel = torch.nn.functional.pad(mel, (0, target_length - T))
    else:
        mel = mel[:, :target_length]
    return mel


class HFAudioDataset(Dataset):
    def __init__(self, split: str, max_samples: int, seq_len: int, max_audio_len: int = 3000,
                 num_frames: int = 4, frame_size: int = 112):
        self.seq_len = seq_len
        self.max_audio_len = max_audio_len

        hf_split = "train" if split == "train" else "test"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ds = load_dataset("lv12/MultiModalDataset", "audioset", split=hf_split, streaming=False)
            self.data = ds.select(range(min(max_samples, len(ds))))

        self._dummy_frames = torch.zeros(num_frames, 3, frame_size, frame_size)
        self._has_visual = torch.tensor(0.0)
        self._has_audio = torch.tensor(1.0)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        text = sample.get("text", "")
        tokenizer = _get_tokenizer()

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            tokens = tokenizer.encode(text, add_special_tokens=False)[: self.seq_len]

        ids = torch.zeros(self.seq_len, dtype=torch.long)
        ids[: len(tokens)] = torch.tensor(tokens, dtype=torch.long)

        audio_data = sample["audio"]
        try:
            if isinstance(audio_data, dict):
                waveform = audio_data.get("array", [])
                sample_rate = audio_data.get("sampling_rate", 16000)
            elif isinstance(audio_data, (list, np.ndarray)):
                waveform = audio_data
                sample_rate = 16000
            else:
                waveform = []
                sample_rate = 16000

            if isinstance(waveform, (list, np.ndarray)) and len(waveform) == 0:
                mel = torch.zeros(80, self.max_audio_len)
            else:
                mel = audio_to_mel_spectrogram(waveform, sample_rate=sample_rate, n_mels=80,
                                               target_length=self.max_audio_len)
        except Exception:
            mel = torch.zeros(80, self.max_audio_len)

        return ids, self._dummy_frames, self._has_visual, mel, self._has_audio
