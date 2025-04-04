from sentence_transformers.util import mine_hard_negatives
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from multiprocessing import cpu_count


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class HardNegativeMiner:
    def __init__(
        self,
        dataset,
        bi_encoder_name="thenlper/gte-base",
        cross_encoder_name="Alibaba-NLP/gte-reranker-modernbert-base",
        max_score=0.8,
    ):

        self.dataset = dataset
        self.bi_encoder = SentenceTransformer(bi_encoder_name, device=DEVICE)
        self.cross_encoder = CrossEncoder(cross_encoder_name, device=DEVICE, model_kwargs={"torch_dtype": "auto"})
        self.max_score = max_score
        self.num_procs = cpu_count() - 1

    def run(self):
        dataset = mine_hard_negatives(
            dataset=self.dataset,
            model=self.bi_encoder,
            cross_encoder=self.cross_encoder,
            anchor_column_name="anchor",
            positive_column_name="document",
            range_min=5,
            range_max=30,
            max_score=self.max_score,
            min_score=0.5,
            margin=0,
            num_negatives=10,
            sampling_strategy="random",
            batch_size=32,
            use_faiss=False,
        )
        dataset = dataset.map(
            {"relevance": 0.9},
            num_proc=self.num_procs,
            remove_columns=["document"],
        )
        return dataset
