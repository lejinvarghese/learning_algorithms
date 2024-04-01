import click
import logging
from datetime import datetime
from tqdm import tqdm

import pandas as pd
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import evaluation

import torch
from torch.utils.data import DataLoader

from preprocess import Preprocessor
from torch.utils.tensorboard import SummaryWriter
from constants import DIRECTORY

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

        logger.info(f"Using device: {self.device}")

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
            if row.get("gain") > 0.0:
                valid_examples[qid]["positive"].add(row.get("product_title"))
            elif row.get("gain") == 0.0:
                valid_examples[qid]["negative"].add(row.get("product_title"))
        self.evaluator = CERerankingEvaluator(valid_examples, name="cross_encoder")

    def compile_model(self):

        self.model = CrossEncoder(
            self.model_name,
            num_labels=1,
            max_length=512,
            default_activation_function=torch.nn.Identity(),
            device=self.device,
        )

    def callback(self, score, epoch, steps):
        self.writer.add_scalar("score", score, (epoch + 1) * steps)

    def fit(self):
        loss = torch.nn.MSELoss()
        self.model.fit(
            train_dataloader=self.train_dataloader,
            loss_fct=loss,
            evaluator=self.evaluator,
            epochs=self.n_epochs,
            evaluation_steps=self.evaluation_steps,
            warmup_steps=self.warmup_steps,
            optimizer_params=self.optimizer_params,
            output_path=f"{DIRECTORY}/models/temp",
            callback=self.callback,
        )
        self.model.save(f"{DIRECTORY}/models/{self.model_name}")

    def train(self):
        self.get_dataloader()
        self.get_evaluator()
        self.compile_model()
        self.fit()


@click.command()
@click.option("--data_version", type=str, default="small_version", help="Data version.")
@click.option("--train_fraction", type=float, default=0.8, help="Train fraction.")
@click.option("--batch_size", type=int, default=16, help="Batch size.")
@click.option(
    "--model_name",
    type=str,
    default="cross-encoder/ms-marco-MiniLM-L-12-v2",
    help="Model name.",
)
@click.option("--n_epochs", type=int, default=1, help="Number of epochs.")
@click.option("--evaluation_steps", type=int, default=1000, help="Steps for evaluation")
@click.option("--warmup_steps", type=int, default=1000, help="Steps for warmup")
@click.option("--learning_rate", type=float, default=8e-6, help="Learning rate")
def main(**kwargs):
    click.secho("Parameters: ", fg="bright_green", bold=True)
    for k, v in kwargs.items():
        click.secho(f"{k}: {v}", fg="bright_cyan")
    # p = Preprocessor(
    #     data_version=kwargs.get("data_version"),
    #     train_fraction=kwargs.get("train_fraction"),
    # )
    # p.process()
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
