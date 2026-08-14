# Runs the model over a dataset in evaluation mode and reports average per-token loss and
# next-token prediction accuracy -- no gradients, no weight updates.
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model, loader: DataLoader, vocab_size: int, device: str, max_batches: int = 10) -> dict:
    """
    Efficient evaluation - only processes max_batches to save memory.
    Full dataset eval would OOM with audio (1500 samples × 120k floats each).
    """
    model.eval()
    total_loss, total_tokens, correct, top5_correct = 0.0, 0, 0, 0

    for batch_idx, (ids, images, has_visual, audio_mel, has_audio) in enumerate(loader):
        if batch_idx >= max_batches:
            break

        ids = ids.to(device)
        images = images.to(device)
        has_visual = has_visual.to(device)
        audio_mel = audio_mel.to(device)
        has_audio = has_audio.to(device)

        logits, _, _ = model(ids, images=images, has_visual=has_visual, audio_mel=audio_mel, has_audio=has_audio)

        # Extract only text logits (skip prepended vision/audio tokens)
        text_len = ids.shape[1]
        num_prepended = logits.shape[1] - text_len
        text_logits = logits[:, num_prepended:, :]

        targets = ids.roll(-1, dims=1)[:, :-1]
        text_logits = text_logits[:, :-1]

        # Mask padding tokens (0s) - they're not real targets
        mask = targets != 0

        if mask.any():
            total_loss += F.cross_entropy(
                text_logits[mask].reshape(-1, vocab_size),
                targets[mask].reshape(-1),
                reduction="sum"
            ).item()
            correct += (text_logits.argmax(-1)[mask] == targets[mask]).sum().item()

            # Top-5 accuracy
            top5_preds = text_logits[mask].topk(5, dim=-1).indices
            top5_correct += (top5_preds == targets[mask].unsqueeze(-1)).any(-1).sum().item()

            total_tokens += mask.sum().item()

        # Free memory after each batch
        del ids, images, has_visual, audio_mel, has_audio, logits, targets, text_logits, mask
        if "cuda" in str(device):
            torch.cuda.empty_cache()

    model.train()
    return {
        "loss": total_loss / total_tokens if total_tokens > 0 else 0.0,
        "accuracy": correct / total_tokens if total_tokens > 0 else 0.0,
        "top5_accuracy": top5_correct / total_tokens if total_tokens > 0 else 0.0,
    }
