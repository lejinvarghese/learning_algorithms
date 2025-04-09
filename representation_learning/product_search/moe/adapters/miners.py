from multiprocessing import cpu_count
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.util import mine_hard_negatives

DATASET_NAME = "lv12/ProductSearchDataset"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class HardNegativeMiner:
    def __init__(
        self,
        dataset,
        bi_encoder_name="thenlper/gte-base",
        # cross_encoder_name="Alibaba-NLP/gte-reranker-modernbert-base",
        max_score=0.8,
        min_score=0.6,
    ):

        self.dataset = dataset
        self.bi_encoder = SentenceTransformer(bi_encoder_name, device=DEVICE)
        # self.cross_encoder = CrossEncoder(cross_encoder_name, device=DEVICE, model_kwargs={"torch_dtype": "auto"})
        self.max_score = max_score
        self.min_score = min_score
        self.num_procs = cpu_count() - 1

    def run(self):
        dataset = mine_hard_negatives(
            dataset=self.dataset,
            model=self.bi_encoder,
            # cross_encoder=self.cross_encoder,
            anchor_column_name="anchor",
            positive_column_name="document",
            range_min=5,
            range_max=10,
            max_score=self.max_score,
            min_score=self.min_score,
            margin=0,
            num_negatives=5,
            sampling_strategy="random",
            batch_size=16,
            use_faiss=False,
            use_multi_process=True
        )
        dataset = dataset.map(
            lambda x: {"relevance": 0.6},
            num_proc=self.num_procs,
            remove_columns=["document"],
        )
        return dataset
