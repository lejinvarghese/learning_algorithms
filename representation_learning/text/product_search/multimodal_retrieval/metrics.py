import torch.nn.functional as F


def compute_metrics(
    anchor,
    positive_text,
    positive_vision,
    negative_text=None,
    negative_vision=None,
):
    pos_text_sim = F.cosine_similarity(anchor, positive_text).mean().item()
    pos_vision_sim = F.cosine_similarity(anchor, positive_vision).mean().item()
    modality_alignment = (
        F.cosine_similarity(positive_text, positive_vision).mean().item()
    )
    if negative_text:
        neg_text_sim = F.cosine_similarity(anchor, negative_text).mean().item()
    if negative_vision:
        neg_vision_sim = F.cosine_similarity(anchor, negative_vision).mean().item()
    # accuracy = (pos_text_sim > pos_vision_sim).float().mean().item()
    return pos_text_sim, pos_vision_sim, modality_alignment
