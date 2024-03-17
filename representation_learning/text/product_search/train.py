from typing import Tuple
import logging
import argparse

import pandas as pd

logging.basicConfig(level=logging.INFO)

DIRECTORY = "data/amazon_search"
SRC_PREFIX = "shopping_queries_dataset"


def preprocess(cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    filters = [("small_version", "==", 1), ("product_locale", "==", "us")]
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

    train = data[data["split"] == "train"]
    test = data[data["split"] == "test"]

    logging.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
    logging.info(test.head())
    if cache:
        train.to_parquet(f"{DIRECTORY}/train.parquet")
        test.to_parquet(f"{DIRECTORY}/test.parquet")
    return train, test


def main(args):
    preprocess(args)


if __name__ == "__main__":
    preprocess()
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size.")
    args = parser.parse_args()
