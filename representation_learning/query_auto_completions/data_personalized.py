"""
Personalized Data loading for Query Auto-Completion

Schema from load_filtered_dataset.py:
- prefix: current typing
- final_search_term: target completion
- historical_searches: list of previous searches in session
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from datasets import load_from_disk
from typing import Optional
import click

from data import ByT5Tokenizer


class PersonalizedQACDataset(Dataset):
    """Dataset for personalized QAC with historical searches"""

    def __init__(
        self,
        hf_dataset,
        tokenizer: ByT5Tokenizer,
        prefix_len: int = 20,
        max_history_len: int = 10,
        max_samples: Optional[int] = None,
        num_negatives: int = 1,
        split_name: str = "unknown",
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.prefix_len = prefix_len
        self.max_history_len = max_history_len
        self.num_negatives = num_negatives
        self.split_name = split_name

        # Apply max_samples limit
        original_size = len(self.hf_dataset)
        if max_samples and len(self.hf_dataset) > max_samples:
            self.hf_dataset = self.hf_dataset.select(range(max_samples))
            click.secho(
                f"  -> {split_name}: Limited to {max_samples} samples (from {original_size})",
                fg="yellow",
            )

        click.secho(
            f"  -> {split_name} dataset size: {len(self.hf_dataset)} raw samples",
            fg="cyan",
        )
        click.secho(
            f"  -> {split_name} with {num_negatives} negative(s): {len(self.hf_dataset) * (1 + num_negatives)} total samples",
            fg="cyan",
        )

        # Cache all final_search_terms for negative sampling (use column directly - much faster)
        self.all_outputs = self.hf_dataset["final_search_term"]
        click.secho(
            f"  -> {split_name} cached {len(self.all_outputs)} outputs for negative sampling",
            fg="green",
        )

        # Show first 3 examples
        click.secho(
            f"\n  First 3 examples from {split_name}:", fg="bright_cyan", bold=True
        )
        for i in range(min(3, len(self.hf_dataset))):
            example = self.hf_dataset[i]
            prefix = example.get("prefix", "")[:50]
            candidate = example.get("final_search_term", "")[:50]
            history = example.get("historical_searches", []) or []
            click.secho(f"    {i+1}. PREFIX: '{prefix}'", fg="white")
            click.secho(f"       CANDIDATE: '{candidate}'", fg="green")
            click.secho(f"       HISTORY: {len(history)} items", fg="yellow")

    def __len__(self):
        return len(self.hf_dataset) * (1 + self.num_negatives)

    def __getitem__(self, idx):
        actual_idx = idx // (1 + self.num_negatives)
        sample_type = idx % (1 + self.num_negatives)

        example = self.hf_dataset[actual_idx]
        prefix_text = example.get("prefix", "")
        output_text = example.get("final_search_term", "")
        history = example.get("historical_searches", []) or []

        prefix_ids = self.tokenizer.encode(prefix_text, self.prefix_len)

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
                candidate_text = output_text if output_text else prefix_text
            label = 0.0

        candidate_ids = self.tokenizer.encode(candidate_text, self.prefix_len)

        # Encode history (keep most recent, truncate to max_history_len)
        history = history[-self.max_history_len :]
        history_ids_list = [self.tokenizer.encode(h, self.prefix_len) for h in history]

        # Pad to max_history_len
        num_hist = len(history_ids_list)
        while len(history_ids_list) < self.max_history_len:
            history_ids_list.append(torch.zeros(self.prefix_len, dtype=torch.long))

        history_ids = torch.stack(history_ids_list)  # [max_history_len, prefix_len]
        history_mask = torch.zeros(self.max_history_len)
        history_mask[:num_hist] = 1.0

        return {
            "prefix_ids": prefix_ids,
            "candidate_ids": candidate_ids,
            "history_ids": history_ids,
            "history_mask": history_mask,
            "label": torch.tensor(label, dtype=torch.float32),
        }


def collate_fn(batch):
    """Collate function for batching - same keys as data.py plus history"""
    return {
        "prefix_ids": torch.stack([item["prefix_ids"] for item in batch]),
        "candidate_ids": torch.stack([item["candidate_ids"] for item in batch]),
        "history_ids": torch.stack([item["history_ids"] for item in batch]),
        "history_mask": torch.stack([item["history_mask"] for item in batch]),
        "labels": torch.stack([item["label"] for item in batch]),
    }


class PersonalizedQACDataModule(pl.LightningDataModule):
    """Data module for personalized QAC dataset"""

    def __init__(
        self,
        dataset_path: str = "./data/amazon_qac_processed",
        tokenizer_name: str = "google/byt5-small",
        batch_size: int = 256,
        prefix_len: int = 20,
        max_history_len: int = 10,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        val_ratio: float = 0.1,
        num_negatives: int = 1,
        num_workers: int = 6,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.prefix_len = prefix_len
        self.max_history_len = max_history_len
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.val_ratio = val_ratio
        self.num_negatives = num_negatives
        self.num_workers = num_workers
        self.tokenizer = None

    def setup(self, stage=None):
        click.secho("\n" + "=" * 60, fg="bright_yellow", bold=True)
        click.secho("PERSONALIZED DATA LOADING", fg="bright_yellow", bold=True)
        click.secho("=" * 60, fg="bright_yellow", bold=True)

        click.secho(
            f"\nLoading dataset from: {self.dataset_path}", fg="cyan", bold=True
        )
        dataset = load_from_disk(self.dataset_path)
        click.secho(f"  Total samples: {len(dataset):,}", fg="green")

        # Check schema
        click.secho(f"\nDataset schema check:", fg="bright_cyan", bold=True)
        sample = dataset[0]
        click.secho(f"  -> Available fields: {list(sample.keys())}", fg="white")
        click.secho(
            f"  -> 'prefix' field present: {'prefix' in sample}",
            fg="green" if "prefix" in sample else "red",
        )
        click.secho(
            f"  -> 'final_search_term' field present: {'final_search_term' in sample}",
            fg="green" if "final_search_term" in sample else "red",
        )
        click.secho(
            f"  -> 'historical_searches' field present: {'historical_searches' in sample}",
            fg="green" if "historical_searches" in sample else "red",
        )

        # Split train/val
        split_data = dataset.train_test_split(test_size=self.val_ratio, seed=42)
        train_data = split_data["train"]
        val_data = split_data["test"]

        click.secho(f"\nData sizes:", fg="bright_cyan", bold=True)
        click.secho(f"  -> Train: {len(train_data)} samples", fg="cyan")
        click.secho(f"  -> Val: {len(val_data)} samples", fg="cyan")

        # Build tokenizer
        self.tokenizer = ByT5Tokenizer(model_name=self.tokenizer_name)

        # Create datasets
        click.secho(f"\nCreating training dataset...", fg="bright_cyan", bold=True)
        self.train_dataset = PersonalizedQACDataset(
            hf_dataset=train_data,
            tokenizer=self.tokenizer,
            prefix_len=self.prefix_len,
            max_history_len=self.max_history_len,
            max_samples=self.max_train_samples,
            num_negatives=self.num_negatives,
            split_name="TRAIN",
        )

        click.secho(f"\nCreating validation dataset...", fg="bright_cyan", bold=True)
        self.val_dataset = PersonalizedQACDataset(
            hf_dataset=val_data,
            tokenizer=self.tokenizer,
            prefix_len=self.prefix_len,
            max_history_len=self.max_history_len,
            max_samples=self.max_val_samples,
            num_negatives=self.num_negatives,
            split_name="VAL",
        )

        click.secho("\n" + "=" * 60, fg="bright_yellow", bold=True)
        click.secho("DATA LOADING COMPLETE", fg="bright_green", bold=True)
        click.secho("=" * 60 + "\n", fg="bright_yellow", bold=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )
