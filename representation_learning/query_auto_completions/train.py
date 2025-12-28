"""
Training script for Search Intention Network using PyTorch Lightning
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import click
from warnings import filterwarnings

from data import QACDataModule
from evaluate import display_evaluation_results
from model import QueryCompletionModel

filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
filterwarnings("ignore", category=UserWarning, module="torch")


class SINLightningModule(pl.LightningModule):
    """PyTorch Lightning module for Query Completion"""

    def __init__(
        self,
        vocab_size=10000,
        embed_dim=128,
        learning_rate=1e-3,
        eval_examples=None,
        use_pretrained_embeddings=False,
        pretrained_model_name="google/byt5-small",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["eval_examples"])

        # Use CNN+Transformer model (IE module from paper)
        self.model = QueryCompletionModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_filters=64,
            num_heads=4,
            num_transformer_layers=2,
            use_pretrained_embeddings=use_pretrained_embeddings,
            pretrained_model_name=pretrained_model_name,
        )

        self.criterion = nn.BCELoss(reduction="mean")
        self.learning_rate = learning_rate
        self.eval_examples = eval_examples or []
        self.label_smoothing = 0.05

    def forward(self, prefix_ids, candidate_ids):
        return self.model(prefix_ids, candidate_ids)

    def training_step(self, batch, batch_idx):
        ctr_scores = self(batch["prefix_ids"], batch["candidate_ids"])

        labels = batch["labels"].squeeze()
        labels_smooth = labels * (1 - self.label_smoothing) + self.label_smoothing / 2

        loss = self.criterion(ctr_scores.squeeze(), labels_smooth)
        predictions = (ctr_scores > 0.5).float()
        accuracy = (predictions.squeeze() == labels).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", accuracy, on_step=True, on_epoch=True)
        self.log("train_pred_mean", ctr_scores.mean(), on_step=False, on_epoch=True)
        self.log("train_pred_std", ctr_scores.std(), on_step=False, on_epoch=True)

        if batch_idx == 0 and self.current_epoch % 5 == 0:
            click.secho(
                f"\n[Epoch {self.current_epoch}] pred=[{ctr_scores.min():.3f}, {ctr_scores.max():.3f}] "
                f"mean={ctr_scores.mean():.3f} | loss={loss:.4f} | acc={accuracy:.4f}",
                fg="yellow",
            )

        return loss

    def validation_step(self, batch, batch_idx):
        ctr_scores = self(batch["prefix_ids"], batch["candidate_ids"])

        loss = self.criterion(ctr_scores.squeeze(), batch["labels"].squeeze())
        predictions = (ctr_scores > 0.5).float()
        accuracy = (predictions.squeeze() == batch["labels"].squeeze()).float().mean()

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", accuracy, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        """Run evaluation on example queries at the end of each validation epoch"""
        if not self.eval_examples or not hasattr(self.trainer.datamodule, "tokenizer"):
            return

        click.secho(
            f"\n📊 Evaluation Examples (Epoch {self.current_epoch}):",
            fg="cyan",
            bold=True,
        )

        tokenizer = self.trainer.datamodule.tokenizer
        for i, example in enumerate(self.eval_examples, 1):
            self.eval()
            with torch.no_grad():
                prefix_ids = (
                    tokenizer.encode(example["prefix"], 20).unsqueeze(0).to(self.device)
                )
                results = []
                for cand in example["candidates"]:
                    cand_ids = tokenizer.encode(cand, 20).unsqueeze(0).to(self.device)
                    score = self(prefix_ids, cand_ids).squeeze().item()
                    results.append((cand, score))
                results.sort(key=lambda x: x[1], reverse=True)
            display_evaluation_results(example["prefix"], results, i)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


@click.command()
@click.option("--epochs", type=int, default=20, help="Number of training epochs")
@click.option("--batch_size", type=int, default=64, help="Batch size")
@click.option("--lr", type=float, default=3e-4, help="Learning rate")
@click.option("--embed_dim", type=int, default=256, help="Embedding dimension")
@click.option(
    "--num_negatives",
    type=int,
    default=1,
    help="Number of negative samples per positive",
)
@click.option("--prefix_len", type=int, default=20, help="Prefix sequence length")
@click.option(
    "--dataset_name",
    type=str,
    default="rexoscare/autocomplete-search-dataset",
    help="HuggingFace dataset name",
)
@click.option(
    "--tokenizer_name",
    type=str,
    default="google/byt5-small",
    help="Tokenizer model name",
)
@click.option(
    "--use_pretrained_embeddings",
    default=True,
    is_flag=True,
    help="Use pretrained ByT5 embeddings",
)
@click.option(
    "--max_train_samples", type=int, default=None, help="Max training samples"
)
@click.option(
    "--max_val_samples", type=int, default=None, help="Max validation samples"
)
@click.option("--val_ratio", type=float, default=0.1, help="Validation split ratio")
@click.option("--gpus", type=int, default=0, help="Number of GPUs (0 for CPU)")
@click.option(
    "--tensorboard_dir",
    type=str,
    default="./lightning_logs",
    help="TensorBoard log directory",
)
@click.option("--experiment_name", type=str, default="sin_qac", help="Experiment name")
@click.option(
    "--eval_examples",
    type=str,
    multiple=True,
    default=[],
    help="Evaluation prefix:candidate1,candidate2,candidate3 (can specify multiple)",
)
def main(
    epochs,
    batch_size,
    lr,
    embed_dim,
    num_negatives,
    prefix_len,
    dataset_name,
    tokenizer_name,
    use_pretrained_embeddings,
    max_train_samples,
    max_val_samples,
    val_ratio,
    gpus,
    tensorboard_dir,
    experiment_name,
    eval_examples,
):
    """Train Search Intention Network for Query Auto-Completion"""

    # Setup data module
    data_module = QACDataModule(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        batch_size=batch_size,
        prefix_len=prefix_len,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        val_ratio=val_ratio,
        num_negatives=num_negatives,
    )
    data_module.setup()

    # Parse evaluation examples
    parsed_eval_examples = []
    if eval_examples:
        for eval_str in eval_examples:
            if ":" in eval_str:
                prefix, candidates_str = eval_str.split(":", 1)
                candidates = [c.strip() for c in candidates_str.split(",")]
                parsed_eval_examples.append(
                    {"prefix": prefix, "candidates": candidates}
                )

    # Default examples if none provided
    if not parsed_eval_examples:
        parsed_eval_examples = [
            {"prefix": "arma", "candidates": ["armadillo", "armageddon", "armor"]},
            {
                "prefix": "casta",
                "candidates": ["castle rock", "castaway", "castor oil"],
            },
        ]

    # Initialize model
    vocab_size = data_module.tokenizer.vocab_size
    model = SINLightningModule(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        learning_rate=lr,
        eval_examples=parsed_eval_examples,
        use_pretrained_embeddings=use_pretrained_embeddings,
        pretrained_model_name=tokenizer_name,
    )

    # Setup trainer
    accelerator = "gpu" if gpus > 0 else "cpu"
    devices = gpus if gpus > 0 else 1

    logger = TensorBoardLogger(tensorboard_dir, name=experiment_name)

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            filename="qac-{epoch:02d}-{val_loss:.4f}",
        ),
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
            min_delta=0.001,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=epochs,
        devices=devices,
        accelerator=accelerator,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=1,
        precision="16-mixed" if gpus > 0 else "32-true",
    )

    # Train
    click.secho("\n🚀 Starting training...", fg="bright_yellow", bold=True)
    trainer.fit(model, data_module)
    click.secho("\n✅ Training completed!\n", fg="bright_green", bold=True)


if __name__ == "__main__":
    main()
