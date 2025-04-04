import re
import click
import logging
import pandas as pd
from typing import Dict

from constants import DIRECTORY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Preprocessor:
    def __init__(self, entity: str, dataset: str):
        self.entity = entity
        self.entity_key = self.entity.split("_")[-1]
        self.dataset = dataset

    def process(self) -> None:
        data = self.read()
        data[f"entity"] = data.apply(self._get_full_matches, axis=1)
        data = data[data.entity.notnull()]

        file_path = f"{DIRECTORY}/{self.dataset}_ner.json"
        data["entity"].to_json(file_path, orient="records")
        logger.info(f"Saved to {file_path}")
        return data

    def read(self) -> pd.DataFrame:
        df = pd.read_parquet(f"{DIRECTORY}/{self.dataset}.parquet")
        cols = ["query", self.entity]
        df = df[(df.gain == 1) & (df[self.entity].notnull())][cols].drop_duplicates()
        df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        return df

    def _get_full_matches(self, row: pd.Series) -> Dict:
        entity = row[self.entity]
        query = row["query"]
        present = re.search(r"\b{}\b".format(re.escape(entity)), query)
        if present:
            start = present.start()
            end = present.end()
            start_token = query[:start].count(" ")
            end_token = query[:end].count(" ")
            return {
                "tokenized_text": query.split(" "),
                "ner": [[start_token, end_token, self.entity_key]],
            }


@click.command()
@click.option("--entity", default="product_brand", help="Entity to preprocess")
@click.option("--dataset", default="train", help="Dataset to preprocess")
def main(entity, dataset):
    p = Preprocessor(entity, dataset)
    data = p.process()
    logger.info(f"Processed {len(data)} examples.")
    logger.info(f"Sample: {data.head()}")


if __name__ == "__main__":
    main()
