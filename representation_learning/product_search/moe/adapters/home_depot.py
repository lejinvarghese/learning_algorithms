from adapters.core import BaseDataset


class HomeDepotDataset(BaseDataset):
    def __init__(self, repo_id="bstds/home_depot", sample_size=None, chunk_size=1000, split="train"):
        super().__init__(repo_id, sample_size, chunk_size, split)
        self.name = "home_depot"
        self.generate_query()
        self.generate_document()

    def generate_document(self):
        self._data = self._data.map(
            lambda row: {
                "document": self.format_document(
                    title=row.get("name"),
                    category=row.get("category"),
                    description=row.get("description"),
                )
            },
            remove_columns=["name", "description", "id", "entity_id"],
            num_proc=self._num_procs,
        )
        self._n_documents = len(set(self._data.unique("document")))

    def generate_triplets(self, threshold=2.3, chunk_index: int = None):
        return super().generate_triplets(threshold=threshold, chunk_index=chunk_index)
