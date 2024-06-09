from multiprocessing import cpu_count
import click
import numpy as np
from datasets import load_dataset, Dataset

DATASET_NAME = "tasksource/esci"
N_SAMPLES = 1_000_000
RANDOM_STATE = 42
N_PROCESS = cpu_count() - 2
np.random.seed(RANDOM_STATE)


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

    def _generate_positives(self, data, threshold):
        pos = data[data["score"] >= threshold][["query", "document"]].copy()
        pos.columns = ["anchor", "positive"]
        click.secho(f"Positives: {pos.shape}", fg="green")
        return pos

    def _generate_negatives(self, data, threshold):
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
        pairs.columns = ["anchor", "document", "score"]
        click.secho(f"Pairs: {pairs.shape}", fg="yellow")
        click.secho(pairs.head(), fg="yellow")
        return Dataset.from_pandas(pairs, preserve_index=False)

    def generate_triplets(self, split, threshold=1.0, n_samples=N_SAMPLES, random_state=RANDOM_STATE):
        if split == "train":
            data = self.train.copy()
        elif split == "test":
            data = self.valid.copy()
        else:
            raise ValueError(f"Invalid split: {split}")
        pos = self._generate_positives(data, threshold)
        neg = self._generate_negatives(data, threshold)
        triplets = pos.merge(neg, on="anchor", how="inner")
        n_samples = min(n_samples, triplets.shape[0])
        triplets = triplets.sample(n_samples, random_state=random_state)[["anchor", "positive", "negative"]]
        click.secho(f"Triplets: {triplets.shape}", fg="yellow")
        click.secho(triplets.head(), fg="yellow")
        return Dataset.from_pandas(triplets, preserve_index=False)

    def generate_ir_datasets(self, split="test", threshold=1.0):
        if split == "train":
            data = self.train.copy()
        elif split == "test":
            data = self.valid.copy()
        else:
            raise ValueError(f"Invalid split: {split}")

        data = data[data["score"] >= threshold]
        query_docs = data["query"].unique()
        n_samples = min(len(query_docs), 1000)
        query_docs = np.random.choice(query_docs, n_samples, replace=False)

        data = data[data["query"].isin(query_docs)]
        corpus_docs = data["document"].unique()

        corpus_ids, query_ids = range(1, len(corpus_docs) + 1), range(1, len(query_docs) + 1)
        corpus, queries = dict(zip(corpus_ids, corpus_docs)), dict(zip(query_ids, query_docs))
        corpus_inverted, queries_inverted = {v: k for k, v in corpus.items()}, {v: k for k, v in queries.items()}

        click.secho(f"Queries: {len(queries)}", fg="yellow")
        click.secho(f"Corpus: {len(corpus)}", fg="yellow")

        relevant_docs = {}

        for q, d in zip(data["query"], data["document"]):
            qid = queries_inverted.get(q, -1)
            corpus_ids = corpus_inverted.get(d, -1)
            if qid not in relevant_docs:
                relevant_docs[qid] = set()
            relevant_docs[qid].add(corpus_ids)

        return queries, corpus, relevant_docs
