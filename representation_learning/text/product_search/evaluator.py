import numpy as np
from typing import Optional
from datetime import datetime
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from torch.utils.tensorboard import SummaryWriter

now = datetime.now().strftime("%Y%m%d%H%M%S")


class CustomCERerankingEvaluator(CERerankingEvaluator):
    """Adds Tensorboard logging to the CERerankingEvaluator class."""

    def __init__(
        self,
        samples,
        at_k: int = 10,
        name: str = "",
        mrr_at_k: Optional[int] = None,
        log_dir: str = None,
    ):
        super().__init__(samples, at_k, name, mrr_at_k)
        self.logs_writer = SummaryWriter(
            log_dir=f"{log_dir}/{now}", filename_suffix=f"{name}"
        )

    def __call__(self, *args, **kwargs):
        mean_mrr = super().__call__(*args, **kwargs)
        self.logs_writer.add_scalar(
            "mrr", mean_mrr, (kwargs.get("epoch") + 1) * kwargs.get("steps")
        )
        return mean_mrr
