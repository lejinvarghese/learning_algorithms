"""
Data loading and tokenization for Query Auto-Completion
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from datasets import load_dataset
from typing import Optional
import click
from transformers import AutoTokenizer


class ByT5Tokenizer:
    """Wrapper for ByT5 byte-level tokenizer"""

    def __init__(self, model_name: str = "google/byt5-small"):
        click.secho(f"\n📝 Loading ByT5 tokenizer from {model_name}...", fg="cyan")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pad_id = self.tokenizer.pad_token_id
        self.vocab_size = len(self.tokenizer)
        click.secho(f"  ✓ Vocabulary size: {self.vocab_size} tokens", fg="green")

    def encode(self, text: str, max_length: int) -> torch.Tensor:
        """Encode text to token ids"""
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return encoded["input_ids"].squeeze(0)

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token ids to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


class QACDataset(Dataset):
    """Dataset for query auto-completion with negative sampling"""

    def __init__(
        self,
        hf_dataset,
        tokenizer: ByT5Tokenizer,
        prefix_len: int = 20,
        max_samples: Optional[int] = None,
        num_negatives: int = 1,
        split_name: str = "unknown",
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.prefix_len = prefix_len
        self.num_negatives = num_negatives
        self.split_name = split_name

        # Apply max_samples limit
        original_size = len(self.hf_dataset)
        if max_samples and len(self.hf_dataset) > max_samples:
            self.hf_dataset = self.hf_dataset.select(range(max_samples))
            click.secho(
                f"  → {split_name}: Limited to {max_samples} samples (from {original_size})",
                fg="yellow",
            )

        click.secho(
            f"  → {split_name} dataset size: {len(self.hf_dataset)} raw samples",
            fg="cyan",
        )
        click.secho(
            f"  → {split_name} with {num_negatives} negative(s): {len(self.hf_dataset) * (1 + num_negatives)} total samples",
            fg="cyan",
        )

        # Cache outputs for negative sampling
        self.all_outputs = [
            item.get("output", "")
            for item in self.hf_dataset
            if item.get("output", "").strip()
        ]
        click.secho(
            f"  → {split_name} cached {len(self.all_outputs)} outputs for negative sampling",
            fg="green",
        )

        # Show first 3 examples
        click.secho(
            f"\n  📋 First 3 examples from {split_name}:", fg="bright_cyan", bold=True
        )
        for i in range(min(3, len(self.hf_dataset))):
            example = self.hf_dataset[i]
            prefix = example.get("input", "")[:50]
            candidate = example.get("output", "")[:50]
            click.secho(f"    {i+1}. PREFIX: '{prefix}'", fg="white")
            click.secho(f"       CANDIDATE: '{candidate}'", fg="green")

    def __len__(self):
        return len(self.hf_dataset) * (1 + self.num_negatives)

    def __getitem__(self, idx):
        actual_idx = idx // (1 + self.num_negatives)
        sample_type = idx % (1 + self.num_negatives)

        example = self.hf_dataset[actual_idx]
        input_text = example.get("input", "")
        output_text = example.get("output", "")

        prefix_ids = self.tokenizer.encode(input_text, self.prefix_len)

        # Positive or negative sample
        if sample_type == 0:
            candidate_text = output_text
            label = 1.0
        else:
            # Negative sampling
            if len(self.all_outputs) > 1:
                for _ in range(10):
                    neg_idx = torch.randint(0, len(self.all_outputs), (1,)).item()
                    candidate_text = self.all_outputs[neg_idx]
                    if candidate_text.lower().strip() != output_text.lower().strip():
                        break
            else:
                candidate_text = output_text if output_text else input_text
            label = 0.0

        candidate_ids = self.tokenizer.encode(candidate_text, self.prefix_len)

        return {
            "prefix_ids": prefix_ids,
            "candidate_ids": candidate_ids,
            "label": torch.tensor(label, dtype=torch.float32),
        }


def collate_fn(batch):
    """Collate function for batching"""
    return {
        "prefix_ids": torch.stack([item["prefix_ids"] for item in batch]),
        "candidate_ids": torch.stack([item["candidate_ids"] for item in batch]),
        "labels": torch.stack([item["label"] for item in batch]),
    }


class QACDataModule(pl.LightningDataModule):
    """Data module for QAC dataset"""

    def __init__(
        self,
        dataset_name: str = "rexoscare/autocomplete-search-dataset",
        tokenizer_name: str = "google/byt5-small",
        batch_size: int = 32,
        prefix_len: int = 20,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        val_ratio: float = 0.1,
        num_negatives: int = 1,
        num_workers: int = 2,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.prefix_len = prefix_len
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.val_ratio = val_ratio
        self.num_negatives = num_negatives
        self.num_workers = num_workers
        self.tokenizer = None

    def setup(self, stage=None):
        click.secho("\n" + "=" * 60, fg="bright_yellow", bold=True)
        click.secho("🔍 DATA LOADING DIAGNOSTICS", fg="bright_yellow", bold=True)
        click.secho("=" * 60, fg="bright_yellow", bold=True)

        click.secho(f"\n📦 Loading dataset: {self.dataset_name}", fg="cyan", bold=True)
        hf_dataset = load_dataset(self.dataset_name)
        click.secho(f"  ✓ Available splits: {list(hf_dataset.keys())}", fg="green")

        # Determine splits
        train_split = "train" if "train" in hf_dataset else list(hf_dataset.keys())[0]
        if "validation" in hf_dataset or "test" in hf_dataset:
            val_split = "validation" if "validation" in hf_dataset else "test"
            train_data = hf_dataset[train_split]
            val_data = hf_dataset[val_split]
            click.secho(
                f"  ✓ Using existing split: {train_split} / {val_split}", fg="green"
            )
        else:
            full_data = hf_dataset[train_split]
            split_data = full_data.train_test_split(test_size=self.val_ratio, seed=42)
            train_data = split_data["train"]
            val_data = split_data["test"]
            click.secho(
                f"  ✓ Created {self.val_ratio:.0%} validation split", fg="green"
            )

        click.secho(f"\n📊 Data sizes:", fg="bright_cyan", bold=True)
        click.secho(f"  → Train: {len(train_data)} samples", fg="cyan")
        click.secho(f"  → Val: {len(val_data)} samples", fg="cyan")

        # Inspect raw dataset schema
        click.secho(f"\n🔎 Dataset schema check:", fg="bright_cyan", bold=True)
        sample = train_data[0]
        click.secho(f"  → Available fields: {list(sample.keys())}", fg="white")
        click.secho(
            f"  → 'input' field present: {'input' in sample}",
            fg="green" if "input" in sample else "red",
        )
        click.secho(
            f"  → 'output' field present: {'output' in sample}",
            fg="green" if "output" in sample else "red",
        )
        if "input" in sample and "output" in sample:
            click.secho(f"\n  📝 Sample raw data:", fg="bright_white", bold=True)
            click.secho(f"     Input:  '{sample['input'][:80]}'", fg="white")
            click.secho(f"     Output: '{sample['output'][:80]}'", fg="green")

        # Build tokenizer
        self.tokenizer = ByT5Tokenizer(model_name=self.tokenizer_name)

        # Create datasets
        click.secho(f"\n🏗️  Creating training dataset...", fg="bright_cyan", bold=True)
        self.train_dataset = QACDataset(
            hf_dataset=train_data,
            tokenizer=self.tokenizer,
            prefix_len=self.prefix_len,
            max_samples=self.max_train_samples,
            num_negatives=self.num_negatives,
            split_name="TRAIN",
        )

        click.secho(f"\n🏗️  Creating validation dataset...", fg="bright_cyan", bold=True)
        self.val_dataset = QACDataset(
            hf_dataset=val_data,
            tokenizer=self.tokenizer,
            prefix_len=self.prefix_len,
            max_samples=self.max_val_samples,
            num_negatives=self.num_negatives,
            split_name="VAL",
        )

        # Test tokenization
        click.secho(f"\n🧪 Tokenization test:", fg="bright_cyan", bold=True)
        test_input = sample.get("input", "")[:20]
        test_output = sample.get("output", "")[:20]
        encoded_input = self.tokenizer.encode(test_input, self.prefix_len)
        encoded_output = self.tokenizer.encode(test_output, self.prefix_len)
        decoded_input = self.tokenizer.decode(encoded_input)
        decoded_output = self.tokenizer.decode(encoded_output)
        click.secho(f"  → Original prefix: '{test_input}'", fg="white")
        click.secho(f"  → Encoded: {encoded_input[:10].tolist()}...", fg="yellow")
        click.secho(f"  → Decoded: '{decoded_input}'", fg="green")
        click.secho(f"  → Original candidate: '{test_output}'", fg="white")
        click.secho(f"  → Encoded: {encoded_output[:10].tolist()}...", fg="yellow")
        click.secho(f"  → Decoded: '{decoded_output}'", fg="green")

        click.secho("\n" + "=" * 60, fg="bright_yellow", bold=True)
        click.secho("✅ DATA LOADING COMPLETE", fg="bright_green", bold=True)
        click.secho("=" * 60 + "\n", fg="bright_yellow", bold=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )
