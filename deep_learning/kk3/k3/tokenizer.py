"""Kimi K3 tokenizer wrapper."""
from transformers import AutoTokenizer


def get_k3_tokenizer():
    """
    Get the official Kimi K3 tokenizer.

    Uses tiktoken-based BPE with 163,840 tokens:
    - 163,328 merges
    - 256 single-byte tokens
    - Special tokens: [BOS], [EOS], [UNK], [PAD]
    """
    return AutoTokenizer.from_pretrained("moonshotai/Kimi-K3", trust_remote_code=True)
