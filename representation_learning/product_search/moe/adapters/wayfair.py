from adapters.core import BaseDataset

FEATURE_COLUMNS = [
    "query",
    "product_id",
    "product_name",
    "product_description",
    "product_features",
    "category hierarchy",
    "label",
]


class WayfairDataset(BaseDataset):
    def __init__(
        self,
        repo_id="napsternxg/wands",
        sample_size=None,
        chunk_size=100,
        split="train",
        cols=FEATURE_COLUMNS,
    ):
        super().__init__(repo_id, sample_size, chunk_size, split)
        self.name = "wayfair"
        self._data = self.load(split, cols)
        self.generate_query()
        self.generate_document()
        self._map_relevance()

    def _map_relevance(self):
        self._data = self._data.map(
            lambda x: {"relevance": float(x["label"])},
            num_proc=self._num_procs,
            remove_columns=["label"],
        )

    def _parse_attributes(self, text):
        """Parse pipe-separated key-value pairs into attributes dictionary.
        Example: "color: red | size: large | material: cotton"
        Returns: {"color": "red", "size": "large", "material": "cotton"}
        """
        if not isinstance(text, str):
            return {}

        attributes = {}
        pairs = [pair.strip() for pair in text.split("|")]

        for pair in pairs:
            try:
                if " : " in pair:
                    key, value = pair.split(" : ", 1)
                    key = key.strip()
                    value = value.strip()
                    print(f"key: {key}, value: {value}", fg="green")
                    if key and value:
                        attributes[key] = value
            except:
                return attributes
        return attributes

    def generate_document(self):
        self._data = self._data.map(
            lambda row: {
                "product_attributes": self._parse_attributes(row.get("product_features", "")),
            },
            num_proc=self._num_procs,
        )
        self._data = self._data.map(
            lambda row: {
                "document": self.format_document(
                    title=row.get("product_name"),
                    description=row.get("product_description"),
                    category=row.get("category hierarchy"),
                    attributes=row.get("product_attributes", {}),
                )
            },
            remove_columns=[
                "product_id",
                "product_name",
                "product_description",
                "product_features",
                "category hierarchy",
                "product_attributes",
            ],
            num_proc=self._num_procs,
        )
        self._n_documents = len(set(self._data.unique("document")))

    def generate_triplets(self, threshold=2, chunk_index: int = None):
        return super().generate_triplets(threshold=threshold, chunk_index=chunk_index)
