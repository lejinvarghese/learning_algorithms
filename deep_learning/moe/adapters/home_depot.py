from adapters.core import BaseDataset


class HomeDepotDataset(BaseDataset):
    def __init__(self, repo_id="bstds/home_depot", sample_size=None, split="train"):
        super().__init__(repo_id, sample_size, split)
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

    def generate_triplets(self, threshold=2.5):
        super().generate_triplets(threshold=threshold)
