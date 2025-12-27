"""
Training script for Search Intention Network using PyTorch Lightning
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import click
from datasets import load_dataset
from collections import Counter
from typing import List, Dict, Optional
from warnings import filterwarnings
from model import SearchIntentionNetwork

# Suppress PyTorch Lightning warnings
filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
filterwarnings("ignore", category=UserWarning, module="torch")


class SimpleTokenizer:
    """Simple character/subword tokenizer for text sequences"""

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.unk_id = 0
        self.pad_id = 1

    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        char_counts = Counter()
        for text in texts:
            char_counts.update(text.lower())
        # Most common chars
        most_common = char_counts.most_common(self.vocab_size - 2)
        for i, (char, _) in enumerate(most_common, start=2):
            self.char_to_id[char] = i
            self.id_to_char[i] = char
        self.char_to_id["<UNK>"] = self.unk_id
        self.char_to_id["<PAD>"] = self.pad_id
        self.id_to_char[self.unk_id] = "<UNK>"
        self.id_to_char[self.pad_id] = "<PAD>"

    def encode(self, text: str, max_length: int) -> torch.Tensor:
        """Encode text to token ids"""
        text = text.lower()
        token_ids = [
            self.char_to_id.get(char, self.unk_id) for char in text[:max_length]
        ]
        # Pad to max_length
        token_ids = token_ids + [self.pad_id] * (max_length - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token ids to text"""
        chars = [self.id_to_char.get(int(id), "<UNK>") for id in token_ids]
        return "".join(chars).replace("<PAD>", "").replace("<UNK>", "")


class QACDataset(Dataset):
    """Dataset for query auto-completion using HuggingFace dataset with negative sampling"""

    def __init__(
        self,
        hf_dataset,
        tokenizer: SimpleTokenizer,
        prefix_len: int = 20,
        seq_len: int = 30,
        num_behaviors: int = 3,
        max_samples: Optional[int] = None,
        num_negatives: int = 1,  # Number of negative samples per positive
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.prefix_len = prefix_len
        self.seq_len = seq_len
        self.num_behaviors = num_behaviors
        self.max_samples = max_samples
        self.num_negatives = num_negatives

        # Limit dataset size if specified
        if max_samples and len(self.hf_dataset) > max_samples:
            self.hf_dataset = self.hf_dataset.select(range(max_samples))

        # Cache all outputs for negative sampling
        self.all_outputs = []
        for item in self.hf_dataset:
            output = item.get("output", "")
            if output and len(output.strip()) > 0:
                self.all_outputs.append(output)
        click.secho(
            f"  Cached {len(self.all_outputs)} outputs for negative sampling", fg="cyan"
        )

    def _find_similar_queries_fast(
        self, query_text: str, current_idx: int, k: int = 10
    ) -> List[int]:
        """Find queries similar to the given query using prefix/character matching (fast, no pre-computation)"""
        query_lower = query_text.lower().strip()
        if not query_lower:
            # If query is empty, return random indices
            candidates = []
            for _ in range(k * 3):  # Sample more than needed
                idx = torch.randint(0, len(self.hf_dataset), (1,)).item()
                if idx != current_idx and idx not in candidates:
                    candidates.append(idx)
                if len(candidates) >= k:
                    break
            return candidates[:k]

        # Find queries that share common prefix or characters (efficient sampling)
        query_chars = set(query_lower)
        query_prefix = query_lower[: min(5, len(query_lower))]  # First few chars

        # Sample subset of dataset for efficiency (don't scan everything)
        num_samples = min(200, len(self.hf_dataset))
        sampled_indices = torch.randperm(len(self.hf_dataset))[:num_samples].tolist()

        scored = []
        for idx in sampled_indices:
            if idx == current_idx:
                continue
            other_example = self.hf_dataset[idx]
            other_text = (
                other_example.get("output", other_example.get("input", ""))
                .lower()
                .strip()
            )

            # Score by: prefix match > char overlap
            if other_text.startswith(query_prefix) and len(query_prefix) > 0:
                score = 2.0  # Strong match
            elif len(query_chars) > 0:
                other_chars = set(other_text)
                if other_chars:
                    char_overlap = len(query_chars & other_chars) / max(
                        len(query_chars | other_chars), 1
                    )
                    score = char_overlap
                else:
                    score = 0.0
            else:
                score = 0.0

            scored.append((score, idx))

        # Sort by score and take top k
        scored.sort(reverse=True)
        similar_indices = [idx for _, idx in scored[:k]]

        # If not enough, pad with random (avoiding current idx)
        while len(similar_indices) < k:
            idx = torch.randint(0, len(self.hf_dataset), (1,)).item()
            if idx != current_idx and idx not in similar_indices:
                similar_indices.append(idx)
            if len(similar_indices) >= len(self.hf_dataset) - 1:
                break

        return similar_indices[:k]

    def __len__(self):
        # Each example generates 1 positive + num_negatives negative samples
        return len(self.hf_dataset) * (1 + self.num_negatives)

    def __getitem__(self, idx):
        # Determine if this is a positive or negative sample
        actual_idx = idx // (1 + self.num_negatives)
        sample_type = idx % (1 + self.num_negatives)  # 0 = positive, >0 = negative

        example = self.hf_dataset[actual_idx]
        input_text = example.get("input", "")
        output_text = example.get("output", "")

        # Encode prefix (always the same for this example)
        prefix_ids = self.tokenizer.encode(input_text, self.prefix_len)

        # For positive sample: use actual output
        # For negative sample: use random output that's different from actual
        if sample_type == 0:
            # POSITIVE EXAMPLE
            candidate_text = output_text
            label = 1.0
        else:
            # NEGATIVE EXAMPLE - sample random output different from actual
            if len(self.all_outputs) > 1:
                # Keep sampling until we get a different output
                max_attempts = 10
                for _ in range(max_attempts):
                    neg_idx = torch.randint(0, len(self.all_outputs), (1,)).item()
                    candidate_text = self.all_outputs[neg_idx]
                    # Make sure it's different from the actual output
                    if candidate_text.lower().strip() != output_text.lower().strip():
                        break
                # If we couldn't find a different one, use a random one anyway
            else:
                candidate_text = output_text if output_text else input_text
            label = 0.0

        candidate_ids = self.tokenizer.encode(candidate_text, self.prefix_len)

        # Create behavior sequences by finding semantically similar queries
        query_for_similarity = output_text if output_text else input_text
        similar_indices = self._find_similar_queries_fast(
            query_for_similarity, current_idx=actual_idx, k=self.num_behaviors
        )

        behavior_sequences = []
        for hist_idx in similar_indices:
            hist_example = self.hf_dataset[hist_idx]
            hist_text = hist_example.get("output", hist_example.get("input", ""))
            seq_ids = self.tokenizer.encode(hist_text, self.seq_len)
            behavior_sequences.append(seq_ids)

        # Create masks (all tokens are valid except padding)
        behavior_masks = [(seq != self.tokenizer.pad_id) for seq in behavior_sequences]

        # Time decay weights (more recent = higher weight)
        time_decays = [
            torch.exp(-torch.arange(self.seq_len).float() / 10.0)
            for _ in range(self.num_behaviors)
        ]

        return {
            "prefix_ids": prefix_ids,
            "candidate_ids": candidate_ids,
            "behavior_sequences": behavior_sequences,
            "behavior_masks": behavior_masks,
            "time_decays": time_decays,
            "label": torch.tensor(label, dtype=torch.float32),
        }


def collate_fn(batch):
    """Collate function for batching"""
    prefix_ids = torch.stack([item["prefix_ids"] for item in batch])
    candidate_ids = torch.stack([item["candidate_ids"] for item in batch])
    behavior_sequences = [
        torch.stack([item["behavior_sequences"][i] for item in batch])
        for i in range(len(batch[0]["behavior_sequences"]))
    ]
    behavior_masks = [
        torch.stack([item["behavior_masks"][i] for item in batch])
        for i in range(len(batch[0]["behavior_masks"]))
    ]
    time_decays = [
        torch.stack([item["time_decays"][i] for item in batch])
        for i in range(len(batch[0]["time_decays"]))
    ]
    labels = torch.stack([item["label"] for item in batch])

    return {
        "prefix_ids": prefix_ids,
        "candidate_ids": candidate_ids,
        "behavior_sequences": behavior_sequences,
        "behavior_masks": behavior_masks,
        "time_decays": time_decays,
        "labels": labels,
    }


class SINLightningModule(pl.LightningModule):
    """PyTorch Lightning module for SIN"""

    def __init__(
        self, vocab_size=10000, embed_dim=128, num_behaviors=3, learning_rate=1e-3
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = SearchIntentionNetwork(
            vocab_size=vocab_size, embed_dim=embed_dim, num_behaviors=num_behaviors
        )
        self.criterion = nn.BCELoss()
        self.learning_rate = learning_rate

    def forward(
        self,
        prefix_ids,
        candidate_ids,
        behavior_sequences,
        behavior_masks=None,
        time_decays=None,
    ):
        return self.model(
            prefix_ids, candidate_ids, behavior_sequences, behavior_masks, time_decays
        )

    def training_step(self, batch, batch_idx):
        prefix_ids = batch["prefix_ids"]
        candidate_ids = batch["candidate_ids"]
        behavior_sequences = batch["behavior_sequences"]
        behavior_masks = batch["behavior_masks"]
        time_decays = batch["time_decays"]
        labels = batch["labels"]

        ctr_scores, _ = self(
            prefix_ids, candidate_ids, behavior_sequences, behavior_masks, time_decays
        )

        # Ensure shapes match
        ctr_scores = ctr_scores.squeeze()
        labels = labels.squeeze()

        loss = self.criterion(ctr_scores, labels)

        # Debug first batch of each epoch
        if batch_idx == 0:
            click.secho(
                f"🔍 Epoch {self.current_epoch} | pred range [{ctr_scores.min():.4f}, {ctr_scores.max():.4f}], "
                f"pred mean: {ctr_scores.mean():.4f} | "
                f"label mean: {labels.mean():.4f} | "
                f"loss: {loss.item():.4f}",
                fg="yellow",
            )

        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            click.secho(f"⚠️  Invalid loss detected: {loss}", fg="red")
            loss = torch.tensor(1.0, requires_grad=True, device=loss.device)

        # Warn if loss is suspiciously low
        if loss.item() < 1e-6 and batch_idx == 0:
            click.secho(
                f"⚠️  Warning: Loss is very small ({loss.item():.6f}). "
                f"This might indicate all labels are identical or model is saturated.",
                fg="yellow",
            )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        prefix_ids = batch["prefix_ids"]
        candidate_ids = batch["candidate_ids"]
        behavior_sequences = batch["behavior_sequences"]
        behavior_masks = batch["behavior_masks"]
        time_decays = batch["time_decays"]
        labels = batch["labels"]

        ctr_scores, _ = self(
            prefix_ids, candidate_ids, behavior_sequences, behavior_masks, time_decays
        )
        ctr_scores = ctr_scores.squeeze()
        labels = labels.squeeze()
        loss = self.criterion(ctr_scores, labels)

        # Calculate accuracy
        predictions = (ctr_scores > 0.5).float()
        accuracy = (predictions == labels).float().mean()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class QACDataModule(pl.LightningDataModule):
    """Data module for QAC dataset"""

    def __init__(
        self,
        dataset_name: str = "rexoscare/autocomplete-search-dataset",
        vocab_size: int = 10000,
        num_behaviors: int = 3,
        batch_size: int = 32,
        prefix_len: int = 20,
        seq_len: int = 30,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        train_split: str = "train",
        val_split: Optional[str] = None,
        val_ratio: float = 0.1,
        num_negatives: int = 1,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.vocab_size = vocab_size
        self.num_behaviors = num_behaviors
        self.batch_size = batch_size
        self.prefix_len = prefix_len
        self.seq_len = seq_len
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.train_split = train_split
        self.val_split = val_split
        self.val_ratio = val_ratio
        self.num_negatives = num_negatives
        self.tokenizer = None

    def setup(self, stage=None):
        click.secho(f"📥 Loading dataset: {self.dataset_name}", fg="cyan")
        try:
            hf_dataset = load_dataset(self.dataset_name)
        except Exception as e:
            click.secho(f"❌ Failed to load dataset: {e}", fg="red")
            raise

        # Determine splits
        if self.val_split and self.val_split in hf_dataset:
            train_data = hf_dataset[self.train_split]
            val_data = hf_dataset[self.val_split]
            click.secho(
                f"✓ Using predefined splits: {self.train_split} / {self.val_split}",
                fg="green",
            )
        else:
            if self.train_split not in hf_dataset:
                self.train_split = list(hf_dataset.keys())[0]
            full_data = hf_dataset[self.train_split]
            split_data = full_data.train_test_split(test_size=self.val_ratio, seed=42)
            train_data = split_data["train"]
            val_data = split_data["test"]
            click.secho(
                f"✓ Split dataset: {len(train_data)} train / {len(val_data)} val",
                fg="green",
            )

        click.secho("🔤 Building tokenizer from training data...", fg="cyan")
        self.tokenizer = SimpleTokenizer(vocab_size=self.vocab_size)
        all_texts = []
        for item in train_data:
            all_texts.append(item.get("input", ""))
            all_texts.append(item.get("output", ""))
        self.tokenizer.build_vocab(all_texts)
        click.secho(f"✓ Vocabulary size: {len(self.tokenizer.char_to_id)}", fg="green")

        # Create datasets
        click.secho("📊 Creating datasets...", fg="cyan")
        self.train_dataset = QACDataset(
            hf_dataset=train_data,
            tokenizer=self.tokenizer,
            prefix_len=self.prefix_len,
            seq_len=self.seq_len,
            num_behaviors=self.num_behaviors,
            max_samples=self.max_train_samples,
            num_negatives=self.num_negatives,
        )
        self.val_dataset = QACDataset(
            hf_dataset=val_data,
            tokenizer=self.tokenizer,
            prefix_len=self.prefix_len,
            seq_len=self.seq_len,
            num_behaviors=self.num_behaviors,
            max_samples=self.max_val_samples,
            num_negatives=self.num_negatives,
        )
        raw_train_size = (
            len(train_data)
            if self.max_train_samples is None
            else min(len(train_data), self.max_train_samples)
        )
        raw_val_size = (
            len(val_data)
            if self.max_val_samples is None
            else min(len(val_data), self.max_val_samples)
        )
        click.secho(
            f"✓ Train samples: {len(self.train_dataset)} ({raw_train_size} base × {1 + self.num_negatives} samples)",
            fg="green",
        )
        click.secho(
            f"✓ Val samples: {len(self.val_dataset)} ({raw_val_size} base × {1 + self.num_negatives} samples)",
            fg="green",
        )

        # Debug: Check label distribution
        sample_labels = [
            self.train_dataset[i]["label"].item()
            for i in range(min(100, len(self.train_dataset)))
        ]
        label_mean = sum(sample_labels) / len(sample_labels)
        click.secho(
            f"ℹ️  Sample label distribution: {label_mean:.2%} positive (should be ~50-90% for this dataset)",
            fg="cyan",
        )

        # Show examples pre and post tokenization
        click.secho("\n📝 Dataset Examples (Raw):", fg="cyan", bold=True)
        for i in range(min(3, len(train_data))):
            example = train_data[i]
            input_text = example.get("input", "")
            output_text = example.get("output", "")

            click.secho(f"\n  Example {i+1}:", fg="yellow")
            click.secho(f"    Input (prefix): '{input_text}'", fg="white")
            click.secho(f"    Output (full):  '{output_text}'", fg="white")

        # Show examples after tokenization and processing
        click.secho("\n📝 Dataset Examples (Processed):", fg="cyan", bold=True)
        for i in range(min(2, len(self.train_dataset))):
            item = self.train_dataset[i]
            click.secho(f"\n  Example {i+1}:", fg="yellow")

            # Decode prefix and candidate to show what was tokenized
            prefix_text = self.tokenizer.decode(item["prefix_ids"])
            candidate_text = self.tokenizer.decode(item["candidate_ids"])
            click.secho(
                f"    Prefix tokens ({len(item['prefix_ids'])}): {item['prefix_ids'][:10].tolist()}...",
                fg="cyan",
            )
            click.secho(f"    Prefix decoded: '{prefix_text[:50]}...'", fg="white")
            click.secho(
                f"    Candidate tokens ({len(item['candidate_ids'])}): {item['candidate_ids'][:10].tolist()}...",
                fg="cyan",
            )
            click.secho(
                f"    Candidate decoded: '{candidate_text[:50]}...'", fg="white"
            )

            # Show behavior sequences
            for j, seq in enumerate(item["behavior_sequences"]):
                seq_text = self.tokenizer.decode(seq)
                click.secho(
                    f"    Behavior {j+1} tokens ({len(seq)}): {seq[:10].tolist()}...",
                    fg="cyan",
                )
                click.secho(
                    f"    Behavior {j+1} decoded: '{seq_text[:50]}...'", fg="white"
                )

            click.secho(
                f"    Label: {item['label'].item():.1f}",
                fg="green" if item["label"].item() > 0 else "red",
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
        )


@click.command()
@click.option("--epochs", type=int, default=10, help="Number of training epochs")
@click.option("--batch_size", type=int, default=32, help="Batch size")
@click.option("--lr", type=float, default=1e-3, help="Learning rate")
@click.option("--embed_dim", type=int, default=128, help="Embedding dimension")
@click.option("--vocab_size", type=int, default=10000, help="Vocabulary size")
@click.option("--num_behaviors", type=int, default=3, help="Number of behavior types")
@click.option(
    "--num_negatives",
    type=int,
    default=3,
    help="Number of negative samples per positive",
)
@click.option("--prefix_len", type=int, default=20, help="Prefix sequence length")
@click.option("--seq_len", type=int, default=30, help="Behavior sequence length")
@click.option(
    "--dataset_name",
    type=str,
    default="rexoscare/autocomplete-search-dataset",
    help="HuggingFace dataset name",
)
@click.option(
    "--max_train_samples",
    type=int,
    default=None,
    help="Max training samples (for testing)",
)
@click.option(
    "--max_val_samples",
    type=int,
    default=None,
    help="Max validation samples (for testing)",
)
@click.option("--val_ratio", type=float, default=0.1, help="Validation split ratio")
@click.option(
    "--gpus",
    type=int,
    default=0,
    help="Number of GPUs (0 for CPU)",
)
def main(
    epochs,
    batch_size,
    lr,
    embed_dim,
    vocab_size,
    num_behaviors,
    num_negatives,
    prefix_len,
    seq_len,
    dataset_name,
    max_train_samples,
    max_val_samples,
    val_ratio,
    gpus,
):
    """Train Search Intention Network for Query Auto-Completion"""
    click.secho("\n🚀 Starting SIN Training", fg="bright_blue", bold=True)
    click.secho("=" * 60, fg="bright_blue")

    # Data module
    click.secho("\n📦 Setting up data module...", fg="cyan")
    click.secho(f"  Using {num_negatives} negative sample(s) per positive", fg="cyan")
    data_module = QACDataModule(
        dataset_name=dataset_name,
        vocab_size=vocab_size,
        num_behaviors=num_behaviors,
        batch_size=batch_size,
        prefix_len=prefix_len,
        seq_len=seq_len,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        val_ratio=val_ratio,
        num_negatives=num_negatives,
    )
    data_module.setup()

    # Get actual vocab size from tokenizer
    actual_vocab_size = len(data_module.tokenizer.char_to_id)
    if actual_vocab_size > vocab_size:
        actual_vocab_size = vocab_size
    click.secho(f"✓ Using vocab size: {actual_vocab_size}", fg="green")

    # Model
    click.secho("\n🏗️  Initializing model...", fg="cyan")
    model = SINLightningModule(
        vocab_size=actual_vocab_size,
        embed_dim=embed_dim,
        num_behaviors=num_behaviors,
        learning_rate=lr,
    )
    total_params = sum(p.numel() for p in model.parameters())
    click.secho(f"✓ Model initialized with {total_params:,} parameters", fg="green")

    # Trainer - determine accelerator
    click.secho("\n⚡ Setting up trainer...", fg="cyan")
    if gpus > 0:
        accelerator = "gpu"
        devices = gpus
        device_name = f"GPU ({gpus} device(s))"
    else:
        accelerator = "cpu"
        devices = 1
        device_name = "CPU"

    click.secho(f"✓ Training on {device_name}", fg="green")
    trainer = pl.Trainer(
        max_epochs=epochs,
        devices=devices,
        accelerator=accelerator,
        callbacks=[ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)],
        enable_progress_bar=True,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,  # Run validation every epoch
        num_sanity_val_steps=2,  # Run 2 validation steps at start to verify it works
    )

    # Train
    click.secho("\n🎯 Starting training...", fg="bright_yellow", bold=True)
    click.secho("=" * 60, fg="bright_yellow")
    try:
        trainer.fit(model, data_module)
        click.secho(
            "\n✅ Training completed successfully!", fg="bright_green", bold=True
        )
    except Exception as e:
        click.secho(f"\n❌ Training failed: {str(e)}", fg="red", bold=True)
        raise


if __name__ == "__main__":
    main()
