from adapters.core import BaseDataset
from click import secho

FEATURE_COLUMNS = [
    "query",
    "product_id",
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
    "product_locale",
    "esci_label",
]

LABEL_MAPPING = {
    "Exact": 3.0,
    "Substitute": 2.0,
    "Complement": 1.0,
    "Irrelevant": 0.0,
}


class AmazonDataset(BaseDataset):
    def __init__(
        self,
        repo_id="tasksource/esci",
        sample_size=None,
        chunk_size=1000,
        split="train",
        cols=FEATURE_COLUMNS,
    ):
        super().__init__(repo_id, sample_size, chunk_size, split, cols)
        self.name = "amazon"
        self._data = self.load(split, cols)
        self._map_relevance()
        self.generate_query()
        self.generate_document()

    def _map_relevance(self):
        self._data = self._data.map(
            lambda x: {"relevance": LABEL_MAPPING.get(x["esci_label"], 0.0)},
            num_proc=self._num_procs,
            remove_columns=["esci_label"],
        )

    def generate_document(self):
        self._data = self._data.filter(lambda row: row.get("product_locale") == "us", num_proc=self._num_procs)
        self._data = self._data.map(
            lambda row: {
                "document": self.format_document(
                    title=row.get("product_title"),
                    description=" ".join(
                        filter(
                            None, [row.get("product_description", "") or "", row.get("product_bullet_point", "") or ""]
                        )
                    ),
                    attributes={
                        "brand": row.get("product_brand"),
                        "color": row.get("product_color"),
                    },
                )
            },
            remove_columns=[
                "product_id",
                "product_title",
                "product_brand",
                "product_color",
                "product_description",
                "product_bullet_point",
                "product_locale",
            ],
            num_proc=self._num_procs,
        )
        self._n_documents = len(set(self._data.unique("document")))
