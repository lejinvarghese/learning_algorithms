"""
Training script for Personalized SIN using PyTorch Lightning
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import click
from warnings import filterwarnings

from data_personalized import PersonalizedQACDataModule
from evaluate import display_evaluation_results
from model_personalized import PersonalizedQueryCompletionModel

filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
filterwarnings("ignore", category=UserWarning, module="torch")


class PersonalizedSINLightningModule(pl.LightningModule):
    """PyTorch Lightning module for Personalized QAC"""

    def __init__(
        self,
        vocab_size=10000,
        embed_dim=128,
        learning_rate=1e-3,
        max_history_len=10,
        eval_examples=None,
        use_pretrained_embeddings=False,
        pretrained_model_name="google/byt5-small",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["eval_examples"])

        self.model = PersonalizedQueryCompletionModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_filters=64,
            num_heads=4,
            num_transformer_layers=2,
            max_history_len=max_history_len,
            use_pretrained_embeddings=use_pretrained_embeddings,
            pretrained_model_name=pretrained_model_name,
        )

        self.criterion = nn.BCELoss(reduction="mean")
        self.learning_rate = learning_rate
        self.eval_examples = eval_examples or []
        self.label_smoothing = 0.05

    def forward(self, prefix_ids, candidate_ids, history_ids=None, history_mask=None):
        return self.model(prefix_ids, candidate_ids, history_ids, history_mask)

    def training_step(self, batch, batch_idx):
        ctr_scores = self(
            batch["prefix_ids"],
            batch["candidate_ids"],
            batch["history_ids"],
            batch["history_mask"],
        )

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
        ctr_scores = self(
            batch["prefix_ids"],
            batch["candidate_ids"],
            batch["history_ids"],
            batch["history_mask"],
        )

        loss = self.criterion(ctr_scores.squeeze(), batch["labels"].squeeze())
        predictions = (ctr_scores > 0.5).float()
        accuracy = (predictions.squeeze() == batch["labels"].squeeze()).float().mean()

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", accuracy, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        """Run evaluation on example queries"""
        if not self.eval_examples or not hasattr(self.trainer.datamodule, "tokenizer"):
            return

        click.secho(
            f"\nEvaluation Examples (Epoch {self.current_epoch}):",
            fg="cyan",
            bold=True,
        )

        tokenizer = self.trainer.datamodule.tokenizer
        prefix_len = self.trainer.datamodule.prefix_len
        max_hist_len = self.trainer.datamodule.max_history_len

        for i, example in enumerate(self.eval_examples, 1):
            self.eval()
            with torch.no_grad():
                prefix_ids = (
                    tokenizer.encode(example["prefix"], prefix_len)
                    .unsqueeze(0)
                    .to(self.device)
                )

                # Encode history if provided
                history = example.get("history", [])
                history_ids_list = [
                    tokenizer.encode(h, prefix_len) for h in history[-max_hist_len:]
                ]
                num_hist = len(history_ids_list)
                while len(history_ids_list) < max_hist_len:
                    history_ids_list.append(torch.zeros(prefix_len, dtype=torch.long))
                history_ids = torch.stack(history_ids_list).unsqueeze(0).to(self.device)
                history_mask = torch.zeros(1, max_hist_len, device=self.device)
                history_mask[0, :num_hist] = 1.0

                results = []
                for cand in example["candidates"]:
                    cand_ids = (
                        tokenizer.encode(cand, prefix_len).unsqueeze(0).to(self.device)
                    )
                    score = (
                        self(prefix_ids, cand_ids, history_ids, history_mask)
                        .squeeze()
                        .item()
                    )
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
@click.option("--batch_size", type=int, default=256, help="Batch size")
@click.option("--lr", type=float, default=3e-4, help="Learning rate")
@click.option("--embed_dim", type=int, default=256, help="Embedding dimension")
@click.option(
    "--num_negatives",
    type=int,
    default=1,
    help="Number of negative samples per positive",
)
@click.option("--prefix_len", type=int, default=20, help="Prefix sequence length")
@click.option("--max_history_len", type=int, default=10, help="Max history length")
@click.option(
    "--dataset_path",
    type=str,
    default="./data/amazon_qac_processed",
    help="Path to processed dataset",
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
    "--mps", is_flag=True, default=False, help="Use Apple Silicon MPS acceleration"
)
@click.option(
    "--num_workers", type=int, default=6, help="DataLoader workers (default 6)"
)
@click.option(
    "--tensorboard_dir",
    type=str,
    default="./lightning_logs",
    help="TensorBoard log directory",
)
@click.option(
    "--experiment_name", type=str, default="sin_personalized", help="Experiment name"
)
def main(
    epochs,
    batch_size,
    lr,
    embed_dim,
    num_negatives,
    prefix_len,
    max_history_len,
    dataset_path,
    tokenizer_name,
    use_pretrained_embeddings,
    max_train_samples,
    max_val_samples,
    val_ratio,
    gpus,
    mps,
    num_workers,
    tensorboard_dir,
    experiment_name,
):
    """Train Personalized SIN for Query Auto-Completion"""

    # Setup data module
    data_module = PersonalizedQACDataModule(
        dataset_path=dataset_path,
        tokenizer_name=tokenizer_name,
        batch_size=batch_size,
        prefix_len=prefix_len,
        max_history_len=max_history_len,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        val_ratio=val_ratio,
        num_negatives=num_negatives,
        num_workers=num_workers,
    )
    data_module.setup()

    # Default eval examples with history
    eval_examples = [
        {
            "prefix": "arma",
            "candidates": ["armadillo", "armageddon", "armor"],
            "history": ["alien vs predator", "avengers"],
        },
        {
            "prefix": "casta",
            "candidates": ["castle rock", "castaway", "castor oil"],
            "history": ["tom hanks movies", "survival films"],
        },
    ]

    # Initialize model
    vocab_size = data_module.tokenizer.vocab_size
    model = PersonalizedSINLightningModule(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        learning_rate=lr,
        max_history_len=max_history_len,
        eval_examples=eval_examples,
        use_pretrained_embeddings=use_pretrained_embeddings,
        pretrained_model_name=tokenizer_name,
    )

    # Setup trainer
    if mps:
        accelerator = "mps"
        devices = 1
    elif gpus > 0:
        accelerator = "gpu"
        devices = gpus
    else:
        accelerator = "cpu"
        devices = 1

    logger = TensorBoardLogger(tensorboard_dir, name=experiment_name)

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            filename="personalized-{epoch:02d}-{val_loss:.4f}",
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
        precision="32-true",  # MPS works best with 32-bit
    )

    # Train
    click.secho("\nStarting training...", fg="bright_yellow", bold=True)
    trainer.fit(model, data_module)
    click.secho("\nTraining completed!\n", fg="bright_green", bold=True)


if __name__ == "__main__":
    main()
