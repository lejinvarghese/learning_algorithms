from typing import Tuple
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

DIRECTORY = "data/amazon_search"
SRC_PREFIX = "shopping_queries_dataset"


def preprocess(cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    filters = [("small_version", "==", 1)]
    judgments = pd.read_parquet(
        f"{DIRECTORY}/{SRC_PREFIX}_examples.parquet", filters=filters
    )
    products = pd.read_parquet(f"{DIRECTORY}/{SRC_PREFIX}_products.parquet")

    data = pd.merge(
        judgments,
        products,
        how="left",
        left_on=["product_locale", "product_id"],
        right_on=["product_locale", "product_id"],
    )

    train = data[data["split"] == "train"]
    test = data[data["split"] == "test"]

    logging.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
    logging.info(test.head())
    if cache:
        train.to_parquet(f"{DIRECTORY}/train.parquet")
        test.to_parquet(f"{DIRECTORY}/test.parquet")
    return train, test


if __name__ == "__main__":
    preprocess()
