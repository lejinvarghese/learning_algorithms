from typing import Tuple, Dict
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from constants import DIRECTORY, RANDOM_STATE

logging.basicConfig(level=logging.INFO)


class Preprocessor:
    def __init__(
        self,
        data_version: str,
        train_fraction: float,
    ):
        self.train_fraction = train_fraction
        self.filters = [(data_version, "==", 1), ("product_locale", "==", "us")]
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

    def process(self) -> None:
        data = self.read()
        splits = self.split(data)
        self.save(splits)

    def read(self):
        file_prefix = "shopping_queries_dataset"
        judgments = pd.read_parquet(
            f"{DIRECTORY}/{file_prefix}_examples.parquet", filters=self.filters
        )
        products = pd.read_parquet(
            f"{DIRECTORY}/{file_prefix}_products.parquet",
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
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = data[data["split"] == "train"][self.cols]
        test = data[data["split"] == "test"][self.cols]

        queries = train["query_id"].unique()
        train_size = int(self.train_fraction * len(queries))
        queries_train, queries_valid = train_test_split(
            queries, train_size=train_size, random_state=RANDOM_STATE
        )

        valid = train[train["query_id"].isin(queries_valid)]
        train = train[train["query_id"].isin(queries_train)]
        logging.info(
            f"Train shape: {train.shape}, Valid shape: {valid.shape}, Test shape: {test.shape}"
        )
        logging.info(train.head())
        return {"train": train, "valid": valid, "test": test}

    def save(self, datasets: Dict[str, pd.DataFrame]) -> None:
        for name, data in datasets.items():
            data.to_parquet(f"{DIRECTORY}/{name}.parquet")
