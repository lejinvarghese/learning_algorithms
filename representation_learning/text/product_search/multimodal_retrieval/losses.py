from torch import nn, cat, stack, tensor


class MultipleNegativesSymmetricRankingLoss(nn.Module):
    def __init__(self, scale=20):
        super().__init__()
        self.similarity_function = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.scale = scale
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def _forward(self, anchor, positive, negative=None):
        if negative:
            candidates = cat([positive, negative], dim=0)
        else:
            candidates = positive
        scores = stack(
            [
                self.similarity_function(a.reshape(1, a.shape[0]), candidates)
                * self.scale
                for a in anchor
            ]
        )
        a_p_scores = scores[:, 0 : positive.size(0)]

        labels = tensor(range(scores.size(0))).to(scores.device)
        forward_loss = self.cross_entropy_loss(scores, labels)
        backward_loss = self.cross_entropy_loss(a_p_scores.transpose(0, 1), labels)
        return (forward_loss + backward_loss) / 2

    def forward(
        self,
        anchor,
        positive,
        reference_anchor,
        reference_positive_text,
        reference_positive_vision,
        negative_text=None,
        negative_vision=None,
    ):
        projection_loss = self._forward(anchor, positive)
        reference_anchor_loss = self._forward(anchor, reference_positive_text)
        reference_positive_text_loss = self._forward(positive, reference_positive_text)
        reference_positive_vision_loss = self._forward(
            positive, reference_positive_vision
        )

        loss = (
            projection_loss
            + reference_anchor_loss
            + reference_positive_text_loss
            + reference_positive_vision_loss
        ) / 2
        return loss
