import json
import re
from abc import ABC
from multiprocessing import cpu_count

from click import secho
from tqdm import trange
from datasets import Dataset, load_dataset, concatenate_datasets

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
        self._split = split
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
            f"Loading data from {self._repo_id} for split {split} using: {self._num_procs} cores",
            fg=(229, 192, 123),
        )
        data = load_dataset(self.repo_id, num_proc=self._num_procs, split=split, columns=cols)
        if self._sample_size is None:
            return data
        else:
            return data.shuffle(seed=RANDOM_STATE).select(range(min(len(data), self._sample_size)))

    def generate_pairs(self):
        pairs = self._data
        source = [self.name] * len(pairs)
        pairs = pairs.add_column("source", source)
        secho(f"Generated {len(pairs)} pairs.", fg="green")
        secho(f"Queries: {self.n_queries}, Documents: {self.n_documents}.", fg="green")
        secho(f"Pairs sample: {pairs[0]}", fg=(229, 192, 123))
        return pairs

    def generate_triplets(self, threshold=3.0):
        # Generate positives and negatives as Hugging Face datasets
        positives = self.generate_positives(threshold=threshold)
        negatives = self.generate_negatives(threshold=threshold)

        # Get unique anchors from both datasets
        positive_anchors = set(positives.unique("anchor"))
        negative_anchors = set(negatives.unique("anchor"))
        common_anchors = positive_anchors & negative_anchors

        # Filter datasets to only include common anchors - do this once
        positives_filtered = positives.filter(lambda x: x["anchor"] in common_anchors)
        negatives_filtered = negatives.filter(lambda x: x["anchor"] in common_anchors)

        # Group by anchor - do this once for the entire dataset
        pos_by_anchor = {}
        neg_by_anchor = {}

        for item in positives_filtered:
            anchor = item["anchor"]
            if anchor not in pos_by_anchor:
                pos_by_anchor[anchor] = []
            pos_by_anchor[anchor].append(item)

        for item in negatives_filtered:
            anchor = item["anchor"]
            if anchor not in neg_by_anchor:
                neg_by_anchor[anchor] = []
            neg_by_anchor[anchor].append(item)

        # Create triplets in batches
        batch_size = 1000
        anchors_list = list(common_anchors)
        all_triplets = []

        for i in trange(0, len(anchors_list), batch_size, desc="Generating triplets", colour="yellow"):
            batch_anchors = anchors_list[i : i + batch_size]
            batch_triplets = []

            for anchor in batch_anchors:
                pos_items = pos_by_anchor.get(anchor, [])
                neg_items = neg_by_anchor.get(anchor, [])

                for pos_item in pos_items:
                    for neg_item in neg_items:
                        # Extract metadata
                        metadata = {}
                        for k, v in pos_item.items():
                            if k not in ["anchor", "positive"]:
                                metadata[f"{k}_positive"] = v
                        for k, v in neg_item.items():
                            if k not in ["anchor", "negative"]:
                                metadata[f"{k}_negative"] = v

                        # Create triplet
                        triplet = {
                            "anchor": anchor,
                            "positive": pos_item["positive"],
                            "negative": neg_item["negative"],
                            "margin": round(pos_item.get("relevance", 0) - neg_item.get("relevance", 0), 2),
                            "source": self.name,
                        }

                        if metadata:
                            triplet["metadata"] = json.dumps(metadata)

                        batch_triplets.append(triplet)

            all_triplets.extend(batch_triplets)

        # Convert to Hugging Face dataset
        triplets = Dataset.from_list(all_triplets)

        secho(f"Generated {len(triplets)} triplets.", fg="green")
        if len(triplets) > 0:
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
