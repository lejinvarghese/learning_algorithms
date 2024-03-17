import os
from typing import Tuple
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


def preprocess(args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    filters = [("small_version", "==", 1), ("product_locale", "==", "us")]
    cols = ["query_id", "query", "product_title", "gain"]
    gain_mapper = {
        "E": 1.0,
        "S": 0.1,
        "C": 0.01,
        "I": 0.0,
    }
    judgments = pd.read_parquet(
        f"{DIRECTORY}/{SRC_PREFIX}_examples.parquet", filters=filters
    )
    products = pd.read_parquet(
        f"{DIRECTORY}/{SRC_PREFIX}_products.parquet", filters=filters[1:]
    )

    data = pd.merge(
        judgments,
        products,
        how="left",
        on="product_id",
    )

    data["gain"] = data["esci_label"].apply(lambda x: gain_mapper[x])
    test = data[data["split"] == "test"][cols]

    queries = data[data["split"] == "train"]["query_id"].unique()
    train_size = int(args.train_fraction * len(queries))
    queries_train, queries_valid = train_test_split(
        queries, train_size=train_size, random_state=args.random_state
    )

    train = data[(data["split"] == "train") & (data["query_id"].isin(queries_train))][
        cols
    ]
    valid = data[(data["split"] == "train") & (data["query_id"].isin(queries_valid))][
        cols
    ]
    logging.info(
        f"Train shape: {train.shape}, Valid shape: {valid.shape}, Test shape: {test.shape}"
    )
    logging.info(train.head())
    if args.cache:
        train.to_parquet(f"{DIRECTORY}/train.parquet")
        valid.to_parquet(f"{DIRECTORY}/valid.parquet")
        test.to_parquet(f"{DIRECTORY}/test.parquet")
    return train, test


def main(args):
    preprocess(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=bool, default=True, help="Cache datasets.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--train_fraction", type=float, default=0.8, help="Train fraction."
    )
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size.")
    args = parser.parse_args()

    main(args)
