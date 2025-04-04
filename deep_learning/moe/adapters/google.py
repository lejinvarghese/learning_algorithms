from click import secho
from datasets import load_dataset
from adapters.core import BaseDataset, RANDOM_STATE

FEATURE_COLUMNS = [
    "query",
    "product_id",
    "title",
    "score_reciprocal",
]


class GoogleDataset(BaseDataset):
    def __init__(
        self,
        repo_id="Marqo/marqo-GS-10M",
        sample_size=None,
        split="train",
        cols=FEATURE_COLUMNS,
    ):
        super().__init__(repo_id, sample_size, split, cols)
        self.name = "google"
        self.generate_query()
        self.generate_document()
        self._map_relevance()

    def _map_relevance(self):
        self._data = self._data.map(
            lambda x: {"relevance": x.get("score_reciprocal", 0.0)},
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
        data = load_dataset(self.repo_id, num_proc=self._num_procs, split=split, columns=cols)
        data = data.filter(lambda row: row.get("product_locale") == "us", num_proc=self._num_procs)
        if self._sample_size is None:
            return data
        else:
            return data.shuffle(seed=RANDOM_STATE).select(range(self._sample_size))

    def generate_document(self):
        self._data = self._data.map(
            lambda row: {"document": self.format_document(title=row.get("product_title"))},
            remove_columns=["product_id"],
            num_proc=self._num_procs,
        )
