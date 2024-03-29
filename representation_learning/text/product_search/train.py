import os
import click
from typing import Tuple, Dict
import logging
import argparse

import pandas as pd
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import evaluation

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

DIRECTORY = "data/amazon_search"
SRC_PREFIX = "shopping_queries_dataset"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


class Preprocessor:
    def __init__(self, directory: str, src_prefix: str):
        self.directory = directory
        self.src_prefix = src_prefix
        self.filters = [("small_version", "==", 1), ("product_locale", "==", "us")]
        self.cols = [
            "query_id",
            "query",
            "product_id",
            "product_title",
            "product_description",
            "product_bullet_point",
            "product_brand",
            "product_color",
            "gain",
        ]
        self.gain_mapper = {
            "E": 1.0,
            "S": 0.1,
            "C": 0.01,
            "I": 0.0,
        }

    def load(self):
        judgments = pd.read_parquet(
            f"{self.directory}/{self.src_prefix}_examples.parquet", filters=self.filters
        )
        products = pd.read_parquet(
            f"{self.directory}/{self.src_prefix}_products.parquet",
            filters=self.filters[1:],
        )

        data = pd.merge(
            judgments,
            products,
            how="left",
            on="product_id",
        )

        data["gain"] = data["esci_label"].apply(lambda x: self.gain_mapper.get(x, 0.0))
        return data

    def split(
        self, data: pd.DataFrame, train_fraction: float = 0.8, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = data[data["split"] == "train"][self.cols]
        test = data[data["split"] == "test"][self.cols]

        queries = train["query_id"].unique()
        train_size = int(train_fraction * len(queries))
        queries_train, queries_valid = train_test_split(
            queries, train_size=train_size, random_state=random_state
        )

        valid = train[train["query_id"].isin(queries_valid)]
        train = train[train["query_id"].isin(queries_train)]
        logging.info(
            f"Train shape: {train.shape}, Valid shape: {valid.shape}, Test shape: {test.shape}"
        )
        logging.info(train.head())
        return {"train": train, "valid": valid, "test": test}

    def save(self, data: Dict[str, pd.DataFrame]) -> None:
        for k, v in data.items():
            v.to_parquet(f"{self.directory}/{k}.parquet")


@click.command()
@click.option("--train_fraction", type=float, default=0.8, help="Train fraction.")
@click.option("--cache", type=bool, default=True, help="Cache datasets.")
@click.option("--random_state", type=int, default=42, help="Random seed.")
def main(train_fraction: float, cache: bool, random_state: int):
    p = Preprocessor(directory=DIRECTORY, src_prefix=SRC_PREFIX)
    data = p.load()
    splits = p.split(data, train_fraction=train_fraction, random_state=random_state)
    if cache:
        p.save(splits)


if __name__ == "__main__":
    main()
