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


class Trainer:
    def __init__(self, batch_size: int, model_name: str, n_epochs: int):
        self.batch_size = batch_size
        self.model_name = model_name
        self.n_epochs = n_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

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
        self.train_dataloader = DataLoader(
            train_examples, shuffle=True, batch_size=self.batch_size, drop_last=True
        )

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
        self.evaluator = CERerankingEvaluator(valid_examples, name="valid")

    def compile_model(self):
        num_labels = 1
        max_length = 512

        self.model = CrossEncoder(
            self.model_name,
            num_labels=num_labels,
            max_length=max_length,
            default_activation_function=torch.nn.Identity(),
            device=self.device,
        )

    def fit(self):
        loss = torch.nn.MSELoss()
        evaluation_steps = 2000
        warmup_steps = 2000
        learning_rate = 8e-5
        self.model.fit(
            train_dataloader=self.train_dataloader,
            loss_fct=loss,
            evaluator=self.evaluator,
            epochs=self.n_epochs,
            evaluation_steps=evaluation_steps,
            warmup_steps=warmup_steps,
            output_path=f"{DIRECTORY}_models_tmp/{self.model_name}",
            optimizer_params={"lr": learning_rate},
        )
        self.model.save(f"{DIRECTORY}_models/{self.model_name}")

    def train(self):
        self.get_dataloader()
        self.get_evaluator()
        self.compile_model()
        self.fit()


@click.command()
@click.option("--data_version", type=str, default="small_version", help="Data version.")
@click.option("--train_fraction", type=float, default=0.8, help="Train fraction.")
@click.option("--batch_size", type=int, default=64, help="Batch size.")
@click.option(
    "--model_name",
    type=str,
    default="cross-encoder/ms-marco-MiniLM-L-12-v2",
    help="Model name.",
)
@click.option("--n_epochs", type=int, default=5, help="Number of epochs.")
def main(**kwargs):
    # p = Preprocessor(
    #     data_version=kwargs.get("data_version"),
    #     train_fraction=kwargs.get("train_fraction"),
    # )
    # p.process()
    t = Trainer(
        batch_size=kwargs.get("batch_size"),
        model_name=kwargs.get("model_name"),
        n_epochs=kwargs.get("n_epochs"),
    )
    t.train()


if __name__ == "__main__":
    main()
