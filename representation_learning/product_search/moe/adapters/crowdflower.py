from click import secho
from adapters.core import BaseDataset


FEATURE_COLUMNS = [
    "query",
    "product_title",
    "product_description",
    "median_relevance",
]


class CrowdFlowerDataset(BaseDataset):
    def __init__(
        self,
        repo_id="napsternxg/kaggle_crowdflower_ecommerce_search_relevance",
        sample_size=None,
        chunk_size=1000,
        split="train",
        cols=FEATURE_COLUMNS,
    ):
        super().__init__(repo_id, sample_size, chunk_size, split, cols)
        self.name = "crowdflower"
        self._data = self.load(split, cols)
        self.generate_query()
        self.generate_document()
        self._map_relevance()

    def _map_relevance(self):
        if self._split == "train":
            self._data = self._data.map(
                lambda x: {"relevance": x.get("median_relevance", 1.0) - 1.0},
                num_proc=self._num_procs,
                remove_columns=["median_relevance"],
            )
        else:
            raise ValueError(f"Skipping {self._split} split due to missing relevance labels")

    def generate_document(self):
        self._data = self._data.map(
            lambda row: {
                "document": self.format_document(
                    title=row.get("product_title"),
                    description=row.get("product_description"),
                )
            },
            remove_columns=["product_title", "product_description"],
            num_proc=self._num_procs,
        )
        self._n_documents = len(set(self._data.unique("document")))
