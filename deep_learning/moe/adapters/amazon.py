from adapters.core import BaseDataset

FEATURE_COLUMNS = [
    "query",
    "product_id",
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
    "esci_label",
]

ESCI_LABEL_MAPPING = {
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
        split="train",
        cols=FEATURE_COLUMNS,
    ):
        super().__init__(repo_id, sample_size, split, cols)
        self.name = "amazon"
        self._map_relevance()
        self.generate_query()
        self.generate_document()

    def _map_relevance(self):
        self._data = self._data.map(
            lambda x: {"relevance": ESCI_LABEL_MAPPING.get(x["esci_label"], 0.0)},
            num_proc=self._num_procs,
            remove_columns=["esci_label"],
        )

    def generate_document(self):
        self._data = self._data.map(
            lambda row: {
                "document": self.format_document(
                    title=row.get("product_title"),
                    description=row.get("product_description"),
                    attributes={
                        "brand": row.get("product_brand"),
                        "color": row.get("product_color"),
                        "details": row.get("product_bullet_point"),
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
            ],
            num_proc=self._num_procs,
        )
