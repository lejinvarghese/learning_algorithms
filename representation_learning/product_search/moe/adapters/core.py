from abc import ABC
from multiprocessing import cpu_count
import json
import re
from click import secho
from datasets import load_dataset, Dataset

RANDOM_STATE = 42


class BaseDataset(ABC):
    def __init__(
        self,
        repo_id: str,
        sample_size: int = None,
        split="train",
        cols: list[str] = None,
    ):
        self._repo_id = repo_id
        self._sample_size = sample_size
        self._num_procs = cpu_count() - 1
        self._data = self.load(split, cols)
        secho(f"Total records loaded: {len(self._data)}", fg="green")

    @property
    def repo_id(self):
        return self._repo_id

    @property
    def data(self):
        return self._data

    @property
    def n_queries(self):
        return self._n_queries

    @property
    def n_documents(self):
        return self._n_documents

    def generate_query(self):
        self._data = self._data.map(
            lambda x: {"query": x["query"].lower()},
            num_proc=self._num_procs,
        )
        self._n_queries = len(set(self._data.unique("query")))

    def generate_document(self):
        pass

    @staticmethod
    def strip_html(text):
        if not isinstance(text, str):
            return ""
        clean = re.compile("<.*?>")
        return re.sub(clean, "", text)

    @staticmethod
    def format_document(**kwargs):
        if kwargs.get("title"):
            template = f"""**product title**: {kwargs.get('title')}\n"""
        else:
            template = """"""
        if kwargs.get("category"):
            template += f"""**product category**: {kwargs.get('category').replace(" / ", " > ")}\n"""
        if kwargs.get("attributes"):
            template += """**product attributes**:\n"""
            for k, v in kwargs.get("attributes").items():
                template += f""" - **{k}**: {v}\n"""

        if kwargs.get("description"):
            template += f"""**product description**: {kwargs.get('description')}"""
        return BaseDataset.strip_html(template.strip().lower())

    def load(self, split: str, cols: list[str] = None):
        secho(
            f"Loading data from {self._repo_id} using: {self._num_procs} cores",
            fg=(229, 192, 123),
        )
        data = load_dataset(self.repo_id, num_proc=self._num_procs, split=split, columns=cols)
        if self._sample_size is None:
            return data
        else:
            return data.shuffle(seed=RANDOM_STATE).select(range(min(len(data), self._sample_size)))

    def generate_pairs(self):
        pairs = self._data
        metadata = [{"source": self.name}] * len(pairs)
        pairs = pairs.add_column("metadata", metadata)
        secho(f"Generated {len(pairs)} pairs.", fg="green")
        secho(f"Queries: {self.n_queries}, Documents: {self.n_documents}.", fg="green")
        secho(f"Pairs sample: {pairs[0]}", fg=(229, 192, 123))
        return pairs

    def generate_triplets(self, threshold=3.0):
        positives = self.generate_positives(threshold=threshold).to_pandas()
        negatives = self.generate_negatives(threshold=threshold).to_pandas()
        triplets = positives.merge(negatives, on="anchor", suffixes=("_positive", "_negative"))
        triplets["margin"] = round(triplets["relevance_positive"] - triplets["relevance_negative"], 2)
        triplets["source"] = self.name

        include_cols = {"anchor", "positive", "negative", "margin"}
        metadata_cols = [col for col in triplets.columns if col not in include_cols]
        triplets["metadata"] = triplets[metadata_cols].apply(lambda x: json.dumps(x.to_dict()), axis=1)
        triplets = triplets.drop(columns=metadata_cols)

        triplets = Dataset.from_pandas(triplets, preserve_index=False)
        secho(f"Generated {len(triplets)} triplets.", fg="green")
        secho(f"Triplets sample: {triplets[0]}", fg=(229, 192, 123))
        return triplets

    def generate_positives(self, threshold):
        pos = self._data.filter(lambda x: x["relevance"] >= threshold).map(
            lambda x: {"anchor": x["query"], "positive": x["document"]},
            num_proc=self._num_procs,
            remove_columns=["query", "document"],
        )
        secho(f"Generated {len(pos)} positives.", fg="green")
        return pos

    def generate_negatives(self, threshold):
        neg = self._data.filter(lambda x: x["relevance"] < threshold).map(
            lambda x: {"anchor": x["query"], "negative": x["document"]},
            num_proc=self._num_procs,
            remove_columns=["query", "document"],
        )
        secho(f"Generated {len(neg)} negatives.", fg="green")
        return neg
