import click
import logging
from tqdm import tqdm

import pandas as pd
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import evaluation

import torch
from torch.utils.data import DataLoader

from preprocess import Preprocessor
from constants import DIRECTORY

logging.basicConfig(level=logging.INFO)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


class Trainer:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def get_dataloader(self):
        train = pd.read_parquet(f"{DIRECTORY}/train.parquet")
        train_examples = []
        for _, row in tqdm(
            train.iterrows(),
            total=len(train),
            desc="Generating training data loader",
            colour="green",
        ):
            train_examples.append(
                InputExample(
                    texts=[row.get("query"), row.get("product_title")],
                    label=float(row.get("gain")),
                )
            )
            train_dataloader = DataLoader(
                train_examples, shuffle=True, batch_size=self.batch_size, drop_last=True
            )
        return train_dataloader

    def get_evaluator(self):
        valid = pd.read_parquet(f"{DIRECTORY}/valid.parquet")
        valid_examples, query_ids = {}, {}
        for _, row in tqdm(
            valid.iterrows(),
            total=len(valid),
            desc="Generating evaluator",
            colour="green",
        ):
            qid = query_ids.get(row.get("query"), len(query_ids))
            if qid == len(query_ids):
                query_ids[row.get("query")] = qid
            if qid not in valid_examples:
                valid_examples[qid] = {
                    "query": row.get("query"),
                    "positive": set(),
                    "negative": set(),
                }
            if row.get("gain") > 0:
                valid_examples[qid]["positive"].add(row.get("product_title"))
            else:
                valid_examples[qid]["negative"].add(row.get("product_title"))
        evaluator = CERerankingEvaluator(valid_examples, name="valid")
        return evaluator


@click.command()
@click.option("--data_version", type=str, default="small_version", help="Data version.")
@click.option("--train_fraction", type=float, default=0.8, help="Train fraction.")
@click.option("--batch_size", type=int, default=32, help="Batch size.")
def main(**kwargs):
    p = Preprocessor(
        data_version=kwargs.get("data_version"),
        train_fraction=kwargs.get("train_fraction"),
    )
    p.run()
    t = Trainer(batch_size=kwargs.get("batch_size"))
    logging.info(type(t.get_dataloader()))
    logging.info(type(t.get_evaluator()))


if __name__ == "__main__":
    main()
