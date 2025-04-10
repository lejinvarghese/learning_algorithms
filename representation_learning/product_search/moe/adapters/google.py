from click import secho
from datasets import Dataset, load_dataset

from adapters.core import RANDOM_STATE, BaseDataset
from adapters.miners import HardNegativeMiner

FEATURE_COLUMNS = [
    "query",
    "product_id",
    "title",
    "score_reciprocal",
]


class GoogleDataset(BaseDataset):
    def __init__(
        self,
        repo_id: str = "Marqo/marqo-GS-10M",
        sample_size: int = None,
        chunk_size: int = 1000,
        split: str = "train",
        cols: list[str] = FEATURE_COLUMNS,
    ):
        super().__init__(repo_id, sample_size, chunk_size, split, cols)
        self.name = "google"
        self._data = self.load(split, cols)
        self.generate_query()
        self.generate_document()
        self._map_relevance()

    def _map_relevance(self):
        self._data = self._data.map(
            lambda x: {"relevance": round(1 + (x.get("score_reciprocal", 0.0) / 100) * 2, 2)},
            num_proc=self._num_procs,
            remove_columns=["score_reciprocal"],
        )

    def load(self, split: str, cols: list[str] = FEATURE_COLUMNS):
        secho(
            f"Loading data from {self._repo_id} using: {self._num_procs} cores",
            fg=(229, 192, 123),
        )
        if split == "train":
            split = "in_domain"
        elif split == "test":
            split = "zero_shot"
        data = load_dataset(self.repo_id, split=split, columns=cols, streaming=True)
        examples = list(data)
        data = Dataset.from_dict({k: [example[k] for example in examples] for k in examples[0].keys()})
        return data

    def generate_document(self):
        self._data = self._data.map(
            lambda row: {"document": self.format_document(title=row.get("title"))},
            remove_columns=["product_id", "title"],
        )
        self._n_documents = len(set(self._data.unique("document")))

    def generate_triplets(self, threshold: float = 1.0, chunk_index: int = None):
        return super().generate_triplets(threshold=threshold, chunk_index=chunk_index)