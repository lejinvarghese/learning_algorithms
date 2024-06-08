from multiprocessing import cpu_count
import click
import numpy as np
from datasets import load_dataset, Dataset

DATASET_NAME = "tasksource/esci"
N_SAMPLES = 1_000_000
RANDOM_STATE = 42
N_PROCESS = cpu_count() - 2


class DataLoader:
    def __init__(
        self,
        dataset_name=DATASET_NAME,
    ):
        self.dataset_name = dataset_name
        self.train, self.valid = self.__download_dataset()

    def __download_dataset(self, n_process=N_PROCESS):
        cols = [
            "query",
            "product_title",
            "product_brand",
            "product_color",
            "product_locale",
            "esci_label",
        ]
        splits = ["train", "test"]
        datasets = []
        for split in splits:
            data = (
                load_dataset(
                    self.dataset_name,
                    columns=cols,
                    split=split,
                    num_proc=n_process,
                )
                .to_pandas()
                .dropna()
            )
            data = data[data.product_locale == "us"]
            click.secho(f"Total records: {split}: {data.shape}", fg="cyan")
            data = self.__generate_scores(data)
            data = self.__generate_queries_and_documents(data)
            datasets.append(data)
        return datasets

    def __generate_scores(self, data):
        conditions = [
            data["esci_label"].isin(["Exact"]),
            data["esci_label"].isin(["Substitute"]),
            data["esci_label"].isin(["Complement"]),
        ]
        grades = [1.0, 0.4, 0.2]
        data["score"] = np.select(conditions, grades, default=0.0)
        return data

    def __generate_queries_and_documents(
        self,
        data,
    ):
        prompts = {"query": "search_query: ", "document": "search_document: "}
        data["document"] = data["product_title"] + ", " + data["product_brand"] + ", " + data["product_color"]
        data["query"] = data["query"].apply(lambda x: f"{prompts.get('query', '')}{x}")
        data["document"] = data["document"].apply(lambda x: f"{prompts.get('document', '')}{x}")
        return data

    def _generate_positives(self, data, threshold=1.0):
        pos = data[data["score"] >= threshold][["query", "document"]].copy()
        pos.columns = ["anchor", "positive"]
        click.secho(f"Positives: {pos.shape}", fg="green")
        return pos

    def _generate_negatives(self, data, threshold=1.0):
        neg = data[data["score"] < threshold][["query", "document"]].copy()
        neg.columns = ["anchor", "negative"]
        click.secho(f"Negatives: {neg.shape}", fg="red")
        return neg

    def generate_pairs(self, split, n_samples=N_SAMPLES, random_state=RANDOM_STATE):
        if split == "train":
            data = self.train.copy()
        elif split == "test":
            data = self.valid.copy()
        else:
            raise ValueError(f"Invalid split: {split}")
        n_samples = min(n_samples, data.shape[0])
        pairs = data.sample(n_samples, random_state=random_state)[["query", "document", "score"]]
        click.secho(f"Pairs: {pairs.shape}", fg="yellow")
        click.secho(pairs.head(), fg="yellow")
        return Dataset.from_pandas(pairs, preserve_index=False)

    def generate_triplets(self, split, n_samples=N_SAMPLES, random_state=RANDOM_STATE):
        if split == "train":
            data = self.train.copy()
        elif split == "test":
            data = self.valid.copy()
        else:
            raise ValueError(f"Invalid split: {split}")
        pos = self._generate_positives(data)
        neg = self._generate_negatives(data)
        triplets = pos.merge(neg, on="anchor", how="inner")
        n_samples = min(n_samples, triplets.shape[0])
        triplets = triplets.sample(n_samples, random_state=random_state)[["anchor", "positive", "negative"]]
        click.secho(f"Triplets: {triplets.shape}", fg="yellow")
        click.secho(triplets.head(), fg="yellow")
        return Dataset.from_pandas(triplets, preserve_index=False)
