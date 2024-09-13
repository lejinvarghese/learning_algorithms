import click
import logging
from datetime import datetime

import pandas as pd
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

import torch
from torch.utils.data import DataLoader

from cross_encoder.preprocess import Preprocessor
from torch.utils.tensorboard import SummaryWriter
from constants import DIRECTORY
from utils import get_pairs, get_triplets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
now = datetime.now().strftime("%Y%m%d%H%M%S")


class Trainer:
    def __init__(
        self,
        batch_size: int,
        model_name: str,
        n_epochs: int,
        evaluation_steps: int,
        warmup_steps: int,
        learning_rate: float,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model_name = model_name
        self.n_epochs = n_epochs
        self.evaluation_steps = evaluation_steps
        self.warmup_steps = warmup_steps
        self.optimizer_params = {"lr": learning_rate}
        self.writer = SummaryWriter(log_dir=f"{DIRECTORY}/logs/{now}")
        self.train_dataloader = self._get_dataloader()
        self.evaluator = self._get_evaluator()
        self.global_step = 0

        logger.info(f"Using device: {self.device}")

    def train(self):
        self.compile()
        self.fit()

    def compile(self):
        self.model = CrossEncoder(
            self.model_name,
            num_labels=1,
            max_length=512,
            default_activation_function=torch.nn.Identity(),
            device=self.device,
        )

    def fit(self):
        loss = torch.nn.CrossEntropyLoss()
        self.model.fit(
            train_dataloader=self.train_dataloader,
            loss_fct=loss,
            evaluator=self.evaluator,
            epochs=self.n_epochs,
            evaluation_steps=self.evaluation_steps,
            warmup_steps=self.warmup_steps,
            scheduler="warmupcosinewithhardrestarts",
            weight_decay=1e-4,
            optimizer_params=self.optimizer_params,
            output_path=f"{DIRECTORY}/models/temp",
            callback=self._callback,
        )
        self.model.save(f"{DIRECTORY}/models/{self.model_name}")

    def _callback(self, score, epoch, steps):
        self.global_step += steps
        self.writer.add_scalar("score", score, self.global_step)

    def _get_dataloader(self):
        train = pd.read_parquet(f"{DIRECTORY}/train.parquet")
        train_examples = get_pairs(train)
        return DataLoader(
            train_examples, shuffle=True, batch_size=self.batch_size, drop_last=True
        )

    def _get_evaluator(self):
        valid = pd.read_parquet(f"{DIRECTORY}/valid.parquet")
        valid_triplets = get_triplets(valid, split="validation")
        return CERerankingEvaluator(valid_triplets, name="cross_encoder")


@click.command()
@click.option("--data_version", type=str, default="small_version", help="Data version.")
@click.option("--train_fraction", type=float, default=0.8, help="Train fraction.")
@click.option("--batch_size", type=int, default=24, help="Batch size.")
@click.option(
    "--model_name",
    type=str,
    default="cross-encoder/ms-marco-MiniLM-L-12-v2",
    help="Model name.",
)
@click.option("--n_epochs", type=int, default=2, help="Number of epochs.")
@click.option("--evaluation_steps", type=int, default=500, help="Steps for evaluation")
@click.option("--warmup_steps", type=int, default=1000, help="Steps for warmup")
@click.option("--learning_rate", type=float, default=1e-5, help="Learning rate")
def main(**kwargs):
    click.secho("Parameters: ", fg="bright_green", bold=True)
    for k, v in kwargs.items():
        click.secho(f"{k}: {v}", fg="bright_cyan")
    p = Preprocessor(
        data_version=kwargs.get("data_version"),
        train_fraction=kwargs.get("train_fraction"),
    )
    p.process()
    t = Trainer(
        batch_size=kwargs.get("batch_size"),
        model_name=kwargs.get("model_name"),
        n_epochs=kwargs.get("n_epochs"),
        evaluation_steps=kwargs.get("evaluation_steps"),
        warmup_steps=kwargs.get("warmup_steps"),
        learning_rate=kwargs.get("learning_rate"),
    )
    t.train()


if __name__ == "__main__":
    main()
