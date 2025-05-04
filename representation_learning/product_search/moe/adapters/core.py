import json
import re
from abc import ABC
from multiprocessing import cpu_count

from click import secho
import random
import pandas as pd

from datasets import Dataset, load_dataset, Features, Value

RANDOM_STATE = 42
random.seed(RANDOM_STATE)

class BaseDataset(ABC):
    def __init__(
        self,
        repo_id: str,
        sample_size: int = None,
        chunk_size: int = 1000,
        split="train"
    ):
        self._repo_id = repo_id
        self._sample_size = sample_size
        self._chunk_size = chunk_size
        self._num_procs = cpu_count() - 1
        self._split = split
        self._data = None
        self._cached_positives = None
        self._cached_negatives = None

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
        secho(f"Generating queries for {self.name} dataset", fg="blue")
        secho(f"Initial dataset size: {len(self._data)}", fg="blue")
        
        self._data = self._data.map(
            lambda x: {"query": x["query"].lower()},
            num_proc=self._num_procs,
        )
        self._unique_queries = list(set(self._data.unique("query")))
        self._n_queries = len(self._unique_queries)
        secho(f"Unique queries: {self._n_queries}", fg="blue")
        
        # Create a single chunk with all queries
        self._query_chunks = {0: self._unique_queries}
        self._max_chunks = 1
        secho(f"Using single chunk with {len(self._unique_queries)} queries", fg="blue")

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
            f"Loading data from {self._repo_id} for split {split} using: {self._num_procs} cores",
            fg=(229, 192, 123),
        )
        data = load_dataset(self._repo_id, num_proc=self._num_procs, split=split, columns=cols)
        secho(f"Total records loaded: {len(data)}", fg="green")
        return data

    def generate_pairs(self):
        pairs = self._data
        source = [self.name] * len(pairs)
        pairs = pairs.add_column("source", source)
        secho(f"Generated {len(pairs)} pairs.", fg="green")
        secho(f"Queries: {self.n_queries}, Documents: {self.n_documents}.", fg="green")
        # secho(f"Pairs sample: {pairs[0]}", fg=(229, 192, 123))
        return pairs

    def generate_triplets(self, threshold=3.0, chunk_index: int = None):
        secho(f"Generating triplets for {self.name} dataset with threshold {threshold}", fg="blue")
        
        positives = self.generate_positives(threshold=threshold).to_pandas()
        secho(f"Generated {len(positives)} positives for {self.name}", fg="blue")
        
        negatives = self.generate_negatives(threshold=threshold).to_pandas()
        secho(f"Generated {len(negatives)} negatives for {self.name}", fg="blue")
        
        if chunk_index is not None:
            chunk_queries = self._query_chunks.get(chunk_index, [])
            secho(f"Filtering for chunk {chunk_index} with {len(chunk_queries)} queries", fg="blue")
            positives = positives[positives["anchor"].isin(chunk_queries)]
            negatives = negatives[negatives["anchor"].isin(chunk_queries)]
            secho(f"After filtering: {len(positives)} positives, {len(negatives)} negatives", fg="blue")
        
        if len(positives) == 0 or len(negatives) == 0:
            secho(f"Not enough data to generate triplets: {len(positives)} positives, {len(negatives)} negatives", fg="red")
            return Dataset.from_dict({
                "anchor": [], "positive": [], "negative": [], 
                "margin": [], "source": [], "metadata": []
            }, features=Features({
                "anchor": Value("string"), "positive": Value("string"), 
                "negative": Value("string"), "margin": Value("float64"),
                "source": Value("string"), "metadata": Value("string")
            }))
        
        if self.name == "google":
            # Create a mapping of anchor to available negatives
            neg_map = negatives.groupby("anchor")["negative"].apply(list).to_dict()
            
            # For each positive, get 5 random negatives from different anchors
            def get_negatives(row):
                anchor = row["anchor"]
                # Get all negatives except those from the same anchor
                available_negs = [neg for a, negs in neg_map.items() 
                                if a != anchor 
                                for neg in negs]
                # Sample 5 random negatives
                if len(available_negs) >= 5:
                    return random.sample(available_negs, 5)
                return available_negs
            
            # Apply the function to each row and explode the results
            triplets = positives.copy()
            triplets["negative"] = triplets.apply(get_negatives, axis=1)
            triplets = triplets.explode("negative")
            
            # Add required columns
            triplets["margin"] = 1.0
            triplets["source"] = self.name
        else:
            triplets = positives.merge(negatives, on="anchor", suffixes=("_positive", "_negative"))
            triplets["margin"] = round(triplets["relevance_positive"] - triplets["relevance_negative"], 2)
            triplets["source"] = self.name
        
        include_cols = {"anchor", "positive", "negative", "margin", "source"}
        metadata_cols = [col for col in triplets.columns if col not in include_cols]
        triplets["metadata"] = triplets[metadata_cols].apply(lambda x: json.dumps(x.to_dict()), axis=1)
        triplets = triplets.drop(columns=metadata_cols)
        
        triplets = Dataset.from_pandas(triplets, preserve_index=False)
        secho(f"Generated {len(triplets)} triplets for {self.name}.", fg="green")
        return triplets

    def generate_positives(self, threshold):
        pos = self._data.filter(lambda x: x["relevance"] >= threshold).map(
            lambda x: {"anchor": x["query"], "positive": x["document"]},
            num_proc=self._num_procs,
            remove_columns=["query", "document"],
        )
        secho(f"Generated {len(pos)} positives for {self.name}", fg="blue")
        return pos

    def generate_negatives(self, threshold):
        neg = self._data.filter(lambda x: x["relevance"] < threshold).map(
            lambda x: {"anchor": x["query"], "negative": x["document"]},
            num_proc=self._num_procs,
            remove_columns=["query", "document"],
        )
        secho(f"Generated {len(neg)} negatives for {self.name}", fg="blue")
        return neg
