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
        skip_size: int = 2000000,
    ):
        super().__init__(repo_id, sample_size, chunk_size, split)
        self.name = "google"
        self._skip_size = skip_size
        self._data = self.load(split, cols)
        self.generate_query()
        self.generate_document()
        self._map_relevance()

    def _map_relevance(self):
        # Process the entire dataset at once since it's already chunked during loading
        self._data = self._data.map(
            lambda x: {"relevance": round(1 + (x.get("score_reciprocal", 0.0) / 100) * 2, 2)},
            num_proc=self._num_procs,
            remove_columns=["score_reciprocal"],
        )

    def load(self, split: str, cols: list[str] = FEATURE_COLUMNS):
        secho(f"Loading data from {self._repo_id}", fg=(229, 192, 123))
        
        if split == "train":
            split = "in_domain"
        elif split == "test":
            split = "zero_shot"
            
        # Load data in streaming mode, skip n records, then take m samples
        sample_size = self._sample_size or 1000  # Default to 1000 if not specified
        skip_size = self._skip_size or 0  # Default to 0 if not specified
        
        data = load_dataset(self.repo_id, split=split, columns=cols, streaming=True)
        if skip_size > 0:
            data = data.skip(skip_size)
        data = data.take(sample_size)
        
        # Convert iterator to Dataset object
        data = Dataset.from_list(list(data))
        secho(f"Loaded {len(data)} total rows (skipped {skip_size})", fg="green")
        return data

    def generate_document(self):
        self._data = self._data.map(
            lambda row: {"document": self.format_document(title=row.get("title"))},
            remove_columns=["product_id", "title"],
        )
        self._n_documents = len(set(self._data.unique("document")))

    def generate_triplets(self, threshold: float = 1.0, chunk_index: int = None):
        return super().generate_triplets(threshold=threshold, chunk_index=chunk_index)

    def generate_negatives(self, threshold):
        # For Google dataset, just pick random documents as negatives
        neg = self._data.map(
            lambda x: {"anchor": x["query"], "negative": x["document"]},
            num_proc=self._num_procs,
            remove_columns=["query", "document"],
        )
        # Shuffle and take a subset for efficiency
        neg = neg.shuffle(seed=RANDOM_STATE).select(range(min(10000, len(neg))))
        secho(f"Generated {len(neg)} random negatives for {self.name}", fg="blue")
        return neg