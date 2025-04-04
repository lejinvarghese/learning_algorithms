from click import secho
from datasets import load_dataset, Dataset
from adapters.core import BaseDataset, RANDOM_STATE
from adapters.negative_miner import HardNegativeMiner

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
        streaming_data = load_dataset(self.repo_id, split=split, columns=cols, streaming=True)
        if cols:
            streaming_data = streaming_data.remove_columns(
                [col for col in streaming_data.column_names if col not in cols]
            )
        if self._sample_size:
            examples = list(streaming_data.take(self._sample_size))
        else:
            examples = list(streaming_data)

        data = Dataset.from_dict({k: [example[k] for example in examples] for k in examples[0].keys()})
        return data.shuffle(seed=RANDOM_STATE)

    def generate_document(self):
        self._data = self._data.map(
            lambda row: {"document": self.format_document(title=row.get("title"))},
            remove_columns=["product_id", "title"],
            num_proc=self._num_procs,
        )
        self._n_documents = len(set(self._data.unique("document")))

    def generate_triplets(self, threshold=1.0):
        return super().generate_triplets(threshold=threshold)

    def generate_negatives(self, threshold=0.8):
        neg = self._data.map(
            lambda x: {"anchor": x["query"]},
            num_proc=self._num_procs,
            remove_columns=["query"],
        )
        neg = HardNegativeMiner(dataset=neg, max_score=threshold).run()
        secho(f"Generated {len(neg)} negatives.", fg="green")
        return neg
