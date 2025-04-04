import torch.nn.functional as F


def compute_metrics(
    anchor,
    positive,
    reference_positive_text,
    reference_positive_vision,
    negative_text=None,
    negative_vision=None,
):
    pos_sim = F.cosine_similarity(anchor, positive).mean().item()
    text_alignment = (
        F.cosine_similarity(positive, reference_positive_text).mean().item()
    )
    image_alignment = (
        F.cosine_similarity(positive, reference_positive_vision).mean().item()
    )
    if negative_text:
        neg_text_sim = F.cosine_similarity(anchor, negative_text).mean().item()
    if negative_vision:
        neg_vision_sim = F.cosine_similarity(anchor, negative_vision).mean().item()
    # accuracy = (pos_text_sim > pos_vision_sim).float().mean().item()
    return pos_sim, text_alignment, image_alignment
