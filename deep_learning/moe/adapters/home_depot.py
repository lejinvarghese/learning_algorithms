import json
from click import secho

from datasets import Dataset
from adapters.core import BaseDataset


class HomeDepotDataset(BaseDataset):
    def __init__(self, repo_id="bstds/home_depot", sample_size=None, split="train"):
        super().__init__(repo_id, sample_size, split)
        self.name = "home_depot"
        self.generate_query()
        self.generate_document()

    def generate_pairs(self):
        self.pairs = self._data
        metadata = [{"source": self.name}] * len(self.pairs)
        self.pairs = self.pairs.add_column("metadata", metadata)
        secho(f"Generated {len(self.pairs)} pairs.", fg="green")
        secho(f"First sample: {self.pairs[0]}", fg="yellow")
        return self.pairs

    def generate_triplets(self, threshold=2.5):
        positives = self._filter_positives(threshold=threshold).to_pandas()
        negatives = self._filter_negatives(threshold=threshold).to_pandas()
        triplets = positives.merge(negatives, on="anchor", suffixes=("_positive", "_negative"))
        triplets["margin"] = round(triplets["relevance_positive"] - triplets["relevance_negative"], 2)
        triplets["source"] = self.name

        include_cols = {"anchor", "positive", "negative", "margin"}
        metadata_cols = [col for col in triplets.columns if col not in include_cols]
        triplets["metadata"] = triplets[metadata_cols].apply(lambda x: json.dumps(x.to_dict()), axis=1)
        triplets = triplets.drop(columns=metadata_cols)

        self.triplets = Dataset.from_pandas(triplets, preserve_index=False)
        secho(f"Generated {len(self.triplets)} triplets.", fg="green")
        secho(f"First sample: {self.triplets[0]}", fg="yellow")
        return self.triplets

    def generate_query(self):
        pass

    def generate_document(self):
        self._data = self._data.map(
            lambda row: {
                "document": self.format_document(
                    title=row.get("name"), category=row.get("category"), description=row.get("description")
                )
            },
            remove_columns=["name", "description", "id", "entity_id"],
            num_proc=self._num_procs,
        )

    def _filter_positives(self, threshold):
        pos = self._data.filter(lambda x: x["relevance"] >= threshold).map(
            lambda x: {"anchor": x["query"], "positive": x["document"]},
            num_proc=self._num_procs,
            remove_columns=["query", "document"],
        )
        secho(f"Generated {len(pos)} positives.", fg="green")
        return pos

    def _filter_negatives(self, threshold):
        neg = self._data.filter(lambda x: x["relevance"] < threshold).map(
            lambda x: {"anchor": x["query"], "negative": x["document"]},
            num_proc=self._num_procs,
            remove_columns=["query", "document"],
        )
        secho(f"Generated {len(neg)} negatives.", fg="green")
        return neg
