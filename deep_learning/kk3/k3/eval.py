# Runs the model over a dataset in evaluation mode and reports average per-token loss and
# next-token prediction accuracy -- no gradients, no weight updates.
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model, loader: DataLoader, vocab_size: int, device: str) -> dict:
    model.eval()
    total_loss, total_tokens, correct = 0.0, 0, 0
    for ids, images, has_visual in loader:
        ids, images, has_visual = ids.to(device), images.to(device), has_visual.to(device)

        logits, _, _ = model(ids, images=images, has_visual=has_visual)
        targets = ids.roll(-1, dims=1)[:, :-1]
        logits = logits[:, :-1]

        total_loss += F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), reduction="sum").item()
        correct += (logits.argmax(-1) == targets).sum().item()
        total_tokens += targets.numel()

    model.train()
    return {"loss": total_loss / total_tokens, "accuracy": correct / total_tokens}
