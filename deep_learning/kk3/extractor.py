#!/usr/bin/env python3
"""
Multi-Dataset Multimodal Extractor
Downloads, processes, and combines multiple datasets into a unified HuggingFace dataset.

Datasets integrated:
- FineWeb-100BT (text, >8192 tokens, high-quality web content)
- COCO 2017 (image+caption, 512x resolution)
- AudioSet (audio+human labels)

Output: Unified dataset with standardized schema, ready for upload to HuggingFace.
Each dataset is extracted with memory-efficient generators and pushed to Hub immediately.
"""
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib

import click
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from PIL import Image
from tqdm import tqdm
import torchaudio
import torchaudio.functional as AF

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


@dataclass
class UnifiedSample:
    """Unified schema for all multimodal samples."""
    id: int  # Simple integer ID
    text: str
    modality: str  # "text", "image", "video", "audio", "video-audio"

    # Visual (only populated for image/video modalities)
    image: Optional[Image.Image] = None
    video_frames: Optional[List[Image.Image]] = None
    num_frames: Optional[int] = None

    # Audio (only populated for audio/video-audio modalities)
    audio_array: Optional[np.ndarray] = None
    audio_sampling_rate: Optional[int] = None

    # Metadata
    duration_sec: Optional[float] = None
    language: str = "en"

    def to_dict(self) -> Dict:
        """Convert to dictionary for HuggingFace datasets."""
        result = {
            "id": self.id,
            "text": self.text,
            "modality": self.modality,
            "language": self.language,
        }

        # Only add fields that are not None (avoid empty columns)
        if self.image is not None:
            result["image"] = self.image
        if self.video_frames is not None:
            result["video_frames"] = self.video_frames
            result["num_frames"] = self.num_frames
        if self.audio_array is not None:
            result["audio"] = {
                "array": self.audio_array,
                "sampling_rate": self.audio_sampling_rate
            }
        if self.duration_sec is not None:
            result["duration_sec"] = self.duration_sec

        return result


class DatasetExtractor:
    """Base class for dataset extractors."""

    def __init__(self, cache_dir: Path, num_workers: int = 16):
        self.cache_dir = cache_dir
        self.num_workers = num_workers
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_config_name(self) -> str:
        """Return the HuggingFace config/subset name for this dataset."""
        raise NotImplementedError

    def extract(self) -> Dict[str, List[UnifiedSample]]:
        """
        Extract and transform dataset to unified schema.

        Returns:
            Dict mapping split_name -> List[UnifiedSample]
            e.g., {"train": [...], "valid": [...], "test": [...]}
        """
        raise NotImplementedError

    def _assign_split(self, idx: int, total: int) -> str:
        """Assign train/valid/test split based on index (80/10/10)."""
        ratio = idx / total
        if ratio < 0.8:
            return "train"
        elif ratio < 0.9:
            return "valid"
        else:
            return "test"


class FineWebExtractor(DatasetExtractor):
    """Extract FineWeb-100BT text dataset (high-quality web text >8192 tokens)."""

    def get_config_name(self) -> str:
        return "fineweb"

    def extract_to_dataset(self) -> DatasetDict:
        """Memory-efficient extraction using generators."""
        click.echo("📚 Loading FineWeb-100BT (>8192 tokens)...")
        try:
            # Load train split with streaming
            ds = load_dataset("NoahEJ/fineweb-sample-100BT_over-8192-tokens", split="train", streaming=True)

            # Take reasonable sample sizes
            train_size = 30000
            valid_size = 5000
            test_size = 2500

            # Generator functions
            def train_generator():
                for idx, row in enumerate(tqdm(ds, total=train_size, desc="FineWeb train")):
                    if idx >= train_size:
                        break
                    yield {
                        "id": idx,
                        "text": row["text"],
                        "modality": "text",
                        "language": "en"
                    }

            def valid_generator():
                # Skip train samples
                for idx, row in enumerate(tqdm(ds, total=train_size + valid_size, desc="FineWeb valid")):
                    if idx < train_size:
                        continue
                    if idx >= train_size + valid_size:
                        break
                    yield {
                        "id": idx - train_size,
                        "text": row["text"],
                        "modality": "text",
                        "language": "en"
                    }

            def test_generator():
                # Skip train + valid samples
                for idx, row in enumerate(tqdm(ds, total=train_size + valid_size + test_size, desc="FineWeb test")):
                    if idx < train_size + valid_size:
                        continue
                    if idx >= train_size + valid_size + test_size:
                        break
                    yield {
                        "id": idx - train_size - valid_size,
                        "text": row["text"],
                        "modality": "text",
                        "language": "en"
                    }

            # Create datasets from generators
            click.echo("  Converting to HF datasets (streaming)...")
            dataset_dict = DatasetDict({
                "train": Dataset.from_generator(train_generator),
                "valid": Dataset.from_generator(valid_generator),
                "test": Dataset.from_generator(test_generator)
            })

            total = sum(len(d) for d in dataset_dict.values())
            click.echo(f"✓ FineWeb: {total} samples (train:{len(dataset_dict['train'])}, valid:{len(dataset_dict['valid'])}, test:{len(dataset_dict['test'])})")

            import gc
            gc.collect()
            return dataset_dict

        except Exception as e:
            click.secho(f"⚠ FineWeb unavailable: {e}", fg="yellow")
            import traceback
            traceback.print_exc()
            return DatasetDict()

    def extract(self) -> Dict[str, List[UnifiedSample]]:
        """Fallback - use extract_to_dataset() instead."""
        return {}


class COCOCaptionExtractor(DatasetExtractor):
    """Extract COCO 2017 512x image-caption-depth dataset."""

    def get_config_name(self) -> str:
        return "coco"

    def extract_to_dataset(self) -> DatasetDict:
        """Memory-efficient extraction using generators."""
        click.echo("🖼️  Loading COCO 2017 512x (memory-efficient streaming)...")
        try:
            # Load train split
            ds = load_dataset("wangherr/coco2017_train_512x_image_caption_depth", split="train")

            # Sample sizes
            total_available = len(ds)
            train_size = min(30000, int(total_available * 0.8))
            valid_size = min(5000, int(total_available * 0.15))
            test_size = min(2500, total_available - train_size - valid_size)

            click.echo(f"  Total available: {total_available}, using {train_size + valid_size + test_size}")

            # Generator functions
            def train_generator():
                for idx in tqdm(range(train_size), desc="COCO train"):
                    row = ds[idx]
                    yield {
                        "id": idx,
                        "text": row["caption"],
                        "modality": "image",
                        "image": row["image"],
                        "language": "en"
                    }

            def valid_generator():
                for idx in tqdm(range(valid_size), desc="COCO valid"):
                    row = ds[train_size + idx]
                    yield {
                        "id": idx,
                        "text": row["caption"],
                        "modality": "image",
                        "image": row["image"],
                        "language": "en"
                    }

            def test_generator():
                for idx in tqdm(range(test_size), desc="COCO test"):
                    row = ds[train_size + valid_size + idx]
                    yield {
                        "id": idx,
                        "text": row["caption"],
                        "modality": "image",
                        "image": row["image"],
                        "language": "en"
                    }

            # Create datasets from generators
            click.echo("  Converting to HF datasets (streaming)...")
            dataset_dict = DatasetDict({
                "train": Dataset.from_generator(train_generator),
                "valid": Dataset.from_generator(valid_generator),
                "test": Dataset.from_generator(test_generator)
            })

            total = sum(len(d) for d in dataset_dict.values())
            click.echo(f"✓ COCO: {total} samples (train:{len(dataset_dict['train'])}, valid:{len(dataset_dict['valid'])}, test:{len(dataset_dict['test'])})")

            # Clean up
            del ds
            import gc
            gc.collect()

            return dataset_dict

        except Exception as e:
            click.secho(f"⚠ COCO unavailable: {e}", fg="yellow")
            import traceback
            traceback.print_exc()
            return DatasetDict()

    def extract(self) -> Dict[str, List[UnifiedSample]]:
        """Fallback - use extract_to_dataset() instead."""
        return {}


class AudioSetExtractor(DatasetExtractor):
    """Extract AudioSet with human labels."""

    def get_config_name(self) -> str:
        return "audioset"

    def extract_to_dataset(self) -> DatasetDict:
        """Memory-efficient extraction using generators."""
        click.echo("🎵 Loading AudioSet (memory-efficient streaming)...")
        try:
            # Load train split
            ds = load_dataset("agkphysics/AudioSet", split="train")

            # Sample sizes
            total_available = len(ds)
            train_size = min(30000, int(total_available * 0.8))
            valid_size = min(5000, int(total_available * 0.15))
            test_size = min(2500, total_available - train_size - valid_size)

            click.echo(f"  Total available: {total_available}, using {train_size + valid_size + test_size}")

            # Generator functions
            def train_generator():
                for idx in tqdm(range(train_size), desc="AudioSet train"):
                    row = ds[idx]
                    # Convert human_labels list to comma-separated string
                    labels = row.get("human_labels", [])
                    label_text = ", ".join(labels) if isinstance(labels, list) else str(labels)

                    audio_data = row.get("audio")
                    yield {
                        "id": idx,
                        "text": label_text,
                        "modality": "audio",
                        "audio": {
                            "array": audio_data["array"] if audio_data else None,
                            "sampling_rate": audio_data["sampling_rate"] if audio_data else None
                        },
                        "language": "en"
                    }

            def valid_generator():
                for idx in tqdm(range(valid_size), desc="AudioSet valid"):
                    row = ds[train_size + idx]
                    labels = row.get("human_labels", [])
                    label_text = ", ".join(labels) if isinstance(labels, list) else str(labels)

                    audio_data = row.get("audio")
                    yield {
                        "id": idx,
                        "text": label_text,
                        "modality": "audio",
                        "audio": {
                            "array": audio_data["array"] if audio_data else None,
                            "sampling_rate": audio_data["sampling_rate"] if audio_data else None
                        },
                        "language": "en"
                    }

            def test_generator():
                for idx in tqdm(range(test_size), desc="AudioSet test"):
                    row = ds[train_size + valid_size + idx]
                    labels = row.get("human_labels", [])
                    label_text = ", ".join(labels) if isinstance(labels, list) else str(labels)

                    audio_data = row.get("audio")
                    yield {
                        "id": idx,
                        "text": label_text,
                        "modality": "audio",
                        "audio": {
                            "array": audio_data["array"] if audio_data else None,
                            "sampling_rate": audio_data["sampling_rate"] if audio_data else None
                        },
                        "language": "en"
                    }

            # Create datasets from generators
            click.echo("  Converting to HF datasets (streaming)...")
            dataset_dict = DatasetDict({
                "train": Dataset.from_generator(train_generator),
                "valid": Dataset.from_generator(valid_generator),
                "test": Dataset.from_generator(test_generator)
            })

            total = sum(len(d) for d in dataset_dict.values())
            click.echo(f"✓ AudioSet: {total} samples (train:{len(dataset_dict['train'])}, valid:{len(dataset_dict['valid'])}, test:{len(dataset_dict['test'])})")

            # Clean up
            del ds
            import gc
            gc.collect()

            return dataset_dict

        except Exception as e:
            click.secho(f"⚠ AudioSet unavailable: {e}", fg="yellow")
            import traceback
            traceback.print_exc()
            return DatasetDict()

    def extract(self) -> Dict[str, List[UnifiedSample]]:
        """Fallback - use extract_to_dataset() instead."""
        return {}


def combine_datasets(
    extractors: List[DatasetExtractor],
    output_dir: Path,
    num_workers: int = 16,
    push_to_hub: bool = False,
    hub_name: str = None
) -> Dict[str, Dict]:
    """
    Extract datasets SEQUENTIALLY and PUSH IMMEDIATELY to avoid RAM overload.
    Each dataset is saved locally, pushed to Hub, then freed from memory.

    Args:
        extractors: List of dataset extractors
        output_dir: Output directory for combined dataset
        num_workers: Number of parallel workers (used within each dataset)
        push_to_hub: Whether to push to HuggingFace Hub immediately after each dataset
        hub_name: HuggingFace Hub dataset name (required if push_to_hub=True)

    Returns:
        Dict mapping config_name -> metadata dict (NOT the full datasets, to save memory)
    """
    click.echo("🔄 Extracting datasets sequentially (to avoid RAM overload)...\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_metadata = {}  # Only store metadata, not the full datasets

    # Process ONE dataset at a time to keep RAM usage manageable
    for idx, extractor in enumerate(extractors, 1):
        config_name = extractor.get_config_name()
        click.echo(f"[{idx}/{len(extractors)}] 📦 Processing {config_name}...")

        try:
            # Use extract_to_dataset() if available (memory-efficient generators)
            # Otherwise fall back to extract() (old list-based method)
            if hasattr(extractor, 'extract_to_dataset'):
                dataset_dict = extractor.extract_to_dataset()
            else:
                # Old method: extract to lists, then convert
                splits_dict = extractor.extract()  # Dict[split_name -> List[UnifiedSample]]

                if not splits_dict:
                    click.secho(f"⚠ {config_name}: No samples extracted", fg="yellow")
                    continue

                # Convert to DatasetDict
                dataset_dict = DatasetDict({
                    split_name: Dataset.from_list([s.to_dict() for s in samples])
                    for split_name, samples in splits_dict.items()
                    if samples
                })

                total = sum(len(d) for d in dataset_dict.values())
                click.echo(f"  Converted {total} samples to HF dataset")

                # Free the splits_dict immediately
                del splits_dict
                import gc
                gc.collect()

            if not dataset_dict:
                click.secho(f"⚠ {config_name}: No datasets created", fg="yellow")
                continue

            # SAVE IMMEDIATELY to local disk
            config_dir = output_dir / config_name
            dataset_dict.save_to_disk(str(config_dir))
            click.secho(f"  ✓ Saved to {config_dir}", fg="green")

            # PUSH IMMEDIATELY to HuggingFace (if requested)
            if push_to_hub and hub_name:
                click.echo(f"  📤 Pushing {config_name} to {hub_name}...")
                try:
                    dataset_dict.push_to_hub(
                        hub_name,
                        config_name=config_name,
                        private=False
                    )
                    click.secho(f"  ✓ Pushed to HuggingFace", fg="green")
                except Exception as e:
                    click.secho(f"  ✗ Push failed: {e}", fg="red")

            # Store metadata (NOT the full dataset)
            total = sum(len(d) for d in dataset_dict.values())
            all_metadata[config_name] = {
                "total": total,
                "splits": {k: len(v) for k, v in dataset_dict.items()}
            }

            # Explicitly FREE MEMORY before next dataset
            del dataset_dict
            import gc
            gc.collect()

            click.echo(f"  💾 Memory freed, ready for next dataset\n")

        except Exception as e:
            click.secho(f"✗ {config_name} failed: {e}", fg="red")
            import traceback
            traceback.print_exc()
            continue

    # Print overall statistics
    click.echo("="*60)
    click.echo(f"📊 Summary: {len(all_metadata)} configs extracted")
    for config_name, metadata in all_metadata.items():
        splits_info = ', '.join(f"{k}:{v}" for k, v in metadata['splits'].items())
        click.echo(f"  ✓ {config_name}: {metadata['total']} samples ({splits_info})")

    click.secho(f"\n✓ All datasets saved to {output_dir}", fg="green")

    return all_metadata


@click.command()
@click.option("--cache-dir", type=click.Path(), default="./data_cache", help="Cache directory for downloads")
@click.option("--output-dir", type=click.Path(), default="./unified_dataset", help="Output directory")
@click.option("--num-workers", type=int, default=16, help="Number of parallel workers")
@click.option("--datasets", type=str, default="fineweb,coco,audioset", help="Comma-separated list: fineweb,coco,audioset")
@click.option("--push-to-hub", is_flag=True, help="Push to HuggingFace Hub after creation")
@click.option("--hub-name", type=str, help="HuggingFace Hub dataset name (e.g., username/dataset-name)")
def main(cache_dir: str, output_dir: str, num_workers: int, datasets: str, push_to_hub: bool, hub_name: str):
    """
    Extract, process, and combine multiple multimodal datasets as HF configs.

    Each dataset becomes a separate config/subset with train/valid/test splits.

    Example:
        python extractor.py --datasets fineweb,coco,audioset --push-to-hub --hub-name lv12/MultiModalDataset
    """
    cache_path = Path(cache_dir)
    output_path = Path(output_dir)

    click.echo("🚀 Multi-Dataset Multimodal Extractor")
    click.echo(f"Cache: {cache_path}")
    click.echo(f"Output: {output_path}")
    click.echo(f"Workers: {num_workers}")
    click.echo(f"Datasets: {datasets}\n")

    # Select extractors
    available_extractors = {
        "fineweb": FineWebExtractor,
        "coco": COCOCaptionExtractor,
        "audioset": AudioSetExtractor,
    }

    selected = [name.strip() for name in datasets.split(",")]
    extractors = []

    for name in selected:
        if name in available_extractors:
            extractors.append(available_extractors[name](cache_path, num_workers))
        else:
            click.secho(f"⚠ Unknown dataset: {name}", fg="yellow")

    if not extractors:
        click.secho("❌ No valid datasets selected", fg="red")
        return

    # Validate push_to_hub requirements
    if push_to_hub and not hub_name:
        click.secho("❌ --hub-name required for push-to-hub", fg="red")
        return

    # Extract all datasets (pushes to Hub immediately if requested)
    # Returns metadata dict (NOT full datasets, to save memory)
    all_metadata = combine_datasets(
        extractors,
        output_path,
        num_workers,
        push_to_hub=push_to_hub,
        hub_name=hub_name
    )

    # Print final summary
    if push_to_hub and all_metadata:
        click.secho(f"\n✅ All {len(all_metadata)} configs uploaded to https://huggingface.co/datasets/{hub_name}", fg="green")
        click.echo(f"\nUsage:")
        click.echo(f"  # Load specific config and split")
        click.echo(f"  ds = load_dataset('{hub_name}', 'imagenet', split='train')")
        click.echo(f"  # Load all splits of a config")
        click.echo(f"  ds = load_dataset('{hub_name}', 'imagenet')")

    click.echo("\n✅ Done!")


if __name__ == "__main__":
    main()
