"""
Preprocess amazon/AmazonQAC dataset:
- Download in streaming mode to handle 400M rows
- Filter by entertainment-related classes using fasttext classifier
- Target classes: music_and_dance, entertainment, celebrity, drama_and_film
- Save filtered dataset locally
"""

import os
import re
import warnings
import numpy as np
from pathlib import Path
import click
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download

# Monkey-patch numpy.array to fix fasttext compatibility with numpy 2.0
_original_array = np.array


def _patched_array(*args, **kwargs):
    if "copy" in kwargs and kwargs["copy"] is False:
        kwargs.pop("copy")
        return np.asarray(*args, **kwargs)
    return _original_array(*args, **kwargs)


np.array = _patched_array

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing")

import fasttext

# Configuration
COLUMNS = ["session_id", "prefixes", "final_search_term", "search_time"]
TARGET_CLASSES = {
    "music_and_dance",
    "entertainment",
    "celebrity",
    "drama_and_film",
    "movie",
    "literature",
    "library",
}
CACHE_DIR = "./data/amazon_qac_cache"
OUTPUT_DIR = "./data/amazon_qac"


def load_classifier():
    """Load fasttext classifier"""
    click.secho("\n📥 Downloading fasttext classifier...", fg="cyan")
    model_path = hf_hub_download(
        "kenhktsui/finefineweb-domain-fasttext-classifier", "model.bin"
    )
    click.secho(f"  ✓ Model downloaded: {model_path}", fg="green")

    click.secho("🔄 Loading classifier...", fg="cyan")
    classifier = fasttext.load_model(model_path)
    click.secho("  ✓ Classifier loaded", fg="green")
    return classifier


def classify_query(classifier, query: str) -> tuple[str, float]:
    """Classify a search query and return (label, score)"""
    # Preprocess: replace newlines and strip
    clean_query = re.sub(r"\n+", " ", query).strip()
    if not clean_query:
        return "unknown", 0.0

    # FastText predict returns (labels, scores)
    labels, scores = classifier.predict(clean_query, k=1)
    label = labels[0].replace("__label__", "")
    score = scores[0]
    return label, score


@click.command()
@click.option(
    "--batch_size", default=10000, help="Number of records to process per batch"
)
@click.option(
    "--max_samples",
    default=None,
    type=int,
    help="Maximum samples to process (None for all)",
)
@click.option(
    "--output_dir", default=OUTPUT_DIR, help="Output directory for filtered dataset"
)
@click.option(
    "--score_threshold", default=0.6, type=float, help="Minimum classification score"
)
def main(batch_size, max_samples, output_dir, score_threshold):
    """Preprocess amazon/AmazonQAC and filter for entertainment classes"""

    click.secho("\n" + "=" * 70, fg="bright_yellow", bold=True)
    click.secho("🎬 AMAZON QAC ENTERTAINMENT FILTER", fg="bright_yellow", bold=True)
    click.secho("=" * 70, fg="bright_yellow", bold=True)

    click.secho(f"\n📋 Configuration:", fg="cyan", bold=True)
    click.secho(f"  Target classes: {', '.join(TARGET_CLASSES)}", fg="white")
    click.secho(f"  Batch size: {batch_size:,}", fg="white")
    click.secho(f"  Score threshold: {score_threshold}", fg="white")
    click.secho(
        f"  Max samples: {max_samples:,}" if max_samples else "  Max samples: All",
        fg="white",
    )
    click.secho(f"  Output: {output_dir}", fg="white")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load classifier
    classifier = load_classifier()

    # Load dataset in streaming mode
    click.secho(
        f"\n📦 Loading amazon/AmazonQAC in streaming mode...", fg="cyan", bold=True
    )
    dataset = load_dataset("amazon/AmazonQAC", split="train", streaming=True)
    dataset = dataset.select_columns(COLUMNS)
    click.secho("  ✓ Dataset loaded", fg="green")

    # Process in batches
    click.secho(f"\n🔄 Processing batches...", fg="cyan", bold=True)

    filtered_records = []
    total_processed = 0
    total_kept = 0
    class_counts = {cls: 0 for cls in TARGET_CLASSES}

    for i, record in enumerate(dataset):
        # Check max samples limit
        if max_samples and total_processed >= max_samples:
            break

        total_processed += 1

        # Classify final_search_term
        final_term = record.get("final_search_term", "")
        if not final_term:
            continue

        label, score = classify_query(classifier, final_term)

        # Keep if in target classes and above threshold
        if label in TARGET_CLASSES and score >= score_threshold:
            filtered_records.append(record)
            total_kept += 1
            class_counts[label] += 1

        # Save batch and report progress
        if len(filtered_records) >= batch_size:
            click.secho(
                f"\n  Batch {total_processed//batch_size}: Processed {total_processed:,} | "
                f"Kept {total_kept:,} ({100*total_kept/total_processed:.2f}%)",
                fg="yellow",
            )
            for cls, count in class_counts.items():
                if count > 0:
                    click.secho(f"    {cls}: {count:,}", fg="green")

            # Save batch
            _save_batch(filtered_records, output_dir, total_kept)
            filtered_records = []

        # Periodic progress update
        if total_processed % 50000 == 0:
            click.secho(
                f"  Progress: {total_processed:,} processed | {total_kept:,} kept | "
                f"Retention: {100*total_kept/total_processed:.3f}%",
                fg="cyan",
            )

    # Save final batch
    if filtered_records:
        click.secho(f"\n  Final batch: {len(filtered_records)} records", fg="yellow")
        _save_batch(filtered_records, output_dir, total_kept)

    # Final report
    click.secho("\n" + "=" * 70, fg="bright_yellow", bold=True)
    click.secho("✅ PROCESSING COMPLETE", fg="bright_green", bold=True)
    click.secho("=" * 70, fg="bright_yellow", bold=True)

    click.secho(f"\n📊 Final Statistics:", fg="cyan", bold=True)
    click.secho(f"  Total processed: {total_processed:,}", fg="white")
    click.secho(f"  Total kept: {total_kept:,}", fg="green")
    click.secho(f"  Retention rate: {100*total_kept/total_processed:.2f}%", fg="yellow")

    click.secho(f"\n📈 Class distribution:", fg="cyan", bold=True)
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / total_kept if total_kept > 0 else 0
        click.secho(f"  {cls}: {count:,} ({pct:.1f}%)", fg="green")

    click.secho(f"\n💾 Output saved to: {output_dir}", fg="cyan")

    os._exit(0)


def _save_batch(records, output_dir, batch_num):
    """Save a batch of records as a separate parquet file"""
    if not records:
        return

    # Create dataset from records
    dataset = Dataset.from_list(records)

    # Save as individual parquet file (zero-padded batch number)
    parquet_dir = f"{output_dir}/parquet_batches"
    Path(parquet_dir).mkdir(parents=True, exist_ok=True)
    parquet_path = f"{parquet_dir}/batch_{batch_num:06d}.parquet"

    dataset.to_parquet(parquet_path)


if __name__ == "__main__":
    main()
