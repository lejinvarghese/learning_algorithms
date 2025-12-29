"""
Load and preprocess parquet data for query auto-completion training

Processes in chunks to avoid memory issues with large datasets.
"""

import click
from datasets import Dataset, concatenate_datasets, load_from_disk
import pandas as pd
from pathlib import Path
import gc


def build_historical_searches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build historical_searches by grouping by session_id.
    For each row, collect all final_search_terms from earlier searches in the same session.
    """
    # Sort by session_id and search_time
    df = df.sort_values(["session_id", "search_time"]).reset_index(drop=True)

    # Build historical searches within each session
    historical = []
    current_session = None
    session_history = []

    for _, row in df.iterrows():
        if row["session_id"] != current_session:
            current_session = row["session_id"]
            session_history = []

        historical.append(session_history.copy())
        session_history.append(row["final_search_term"])

    df["historical_searches"] = historical
    return df


def explode_prefixes(batch):
    """Explode prefixes array into individual rows, deduplicating prefixes"""
    out_prefix = []
    out_final = []
    out_hist = []

    for i in range(len(batch["prefixes"])):
        prefixes = batch["prefixes"][i]
        if not prefixes:
            continue

        final_search_term = batch["final_search_term"][i]
        historical = batch["historical_searches"][i] or []

        seen = set()
        unique_prefixes = []
        for p in prefixes:
            if p not in seen:
                seen.add(p)
                unique_prefixes.append(p)

        for prefix in unique_prefixes:
            out_prefix.append(prefix)
            out_final.append(final_search_term)
            out_hist.append(historical)

    return {
        "prefix": out_prefix,
        "final_search_term": out_final,
        "historical_searches": out_hist,
    }


def filter_prefix_not_equal_target(example):
    """Remove rows where prefix equals final_search_term"""
    return example["prefix"] != example["final_search_term"]


def process_chunk(df_chunk: pd.DataFrame, chunk_idx: int, output_dir: Path) -> int:
    """Process a single chunk and save to disk. Returns number of records."""
    click.secho(
        f"\n  Chunk {chunk_idx}: Converting to Dataset ({len(df_chunk):,} rows)...",
        fg="white",
    )

    dataset = Dataset.from_pandas(df_chunk)

    click.secho(f"  Chunk {chunk_idx}: Exploding prefixes...", fg="white")
    dataset = dataset.map(
        explode_prefixes,
        remove_columns=dataset.column_names,
        batched=True,
        batch_size=10000,
    )

    click.secho(f"  Chunk {chunk_idx}: Filtering...", fg="white")
    dataset = dataset.filter(filter_prefix_not_equal_target)

    chunk_path = output_dir / f"chunk_{chunk_idx:03d}"
    dataset.save_to_disk(str(chunk_path))
    click.secho(
        f"  Chunk {chunk_idx}: Saved {len(dataset):,} records to {chunk_path.name}",
        fg="green",
    )

    num_records = len(dataset)
    del dataset
    gc.collect()

    return num_records


@click.command()
@click.option(
    "--input_dir",
    default="./data/amazon_qac",
    help="Input directory with parquet batches",
)
@click.option(
    "--output_path",
    default="./data/amazon_qac_processed",
    help="Output path for final dataset",
)
@click.option(
    "--single_file", default=None, help="Load a single parquet file for experimentation"
)
@click.option(
    "--chunk_size", default=500_000, type=int, help="Chunk size for processing"
)
@click.option(
    "--undersample",
    default=None,
    type=int,
    help="Randomly undersample to N records (e.g., 5000000)",
)
@click.option(
    "--min_prefix_len", default=2, type=int, help="Minimum prefix length to keep"
)
def main(input_dir, output_path, single_file, chunk_size, undersample, min_prefix_len):
    """Load and preprocess parquet data in chunks"""

    parquet_dir = f"{input_dir}/parquet_batches"
    temp_dir = Path(output_path + "_chunks")
    temp_dir.mkdir(parents=True, exist_ok=True)

    click.secho("\n" + "=" * 60, fg="bright_yellow", bold=True)
    click.secho(
        "LOADING AND PREPROCESSING DATASET (CHUNKED)", fg="bright_yellow", bold=True
    )
    click.secho("=" * 60, fg="bright_yellow", bold=True)
    click.secho(f"Chunk size: {chunk_size:,}", fg="cyan")

    # Check if directory exists
    if not Path(parquet_dir).exists():
        click.secho(f"\nDirectory not found: {parquet_dir}", fg="red", bold=True)
        return

    # Determine which files to load
    if single_file:
        parquet_path = Path(parquet_dir) / single_file
        if not parquet_path.exists():
            click.secho(f"File not found: {parquet_path}", fg="red", bold=True)
            return
        parquet_files = [parquet_path]
    else:
        parquet_files = sorted(Path(parquet_dir).glob("*.parquet"))
        click.secho(f"\nFound {len(parquet_files)} parquet batch files", fg="cyan")
        if not parquet_files:
            click.secho("No parquet files found!", fg="red", bold=True)
            return

    # Load parquet files
    click.secho("\nLoading data...", fg="cyan")
    try:
        dfs = []
        for pf in parquet_files:
            dfs.append(pd.read_parquet(pf))
            click.secho(f"  Loaded {pf.name}: {len(dfs[-1]):,} rows", fg="white")
        df = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()
        click.secho(f"Total loaded: {len(df):,} rows", fg="green")
    except Exception as e:
        click.secho(f"[ERROR] Failed to load parquet: {e}", fg="red")
        import traceback

        traceback.print_exc()
        return

    # Remove duplicates
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["session_id", "final_search_term", "search_time"])
    if before_dedup > len(df):
        click.secho(f"Removed {before_dedup - len(df):,} duplicate rows", fg="yellow")

    # Build historical searches (needs full data for session grouping)
    click.secho("\nBuilding historical searches from session data...", fg="cyan")
    df = build_historical_searches(df)
    non_empty_hist = (df["historical_searches"].apply(len) > 0).sum()
    click.secho(
        f"Rows with historical searches: {non_empty_hist:,} ({100*non_empty_hist/len(df):.1f}%)",
        fg="green",
    )

    # Process in chunks
    total_rows = len(df)
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    click.secho(
        f"\nProcessing {total_rows:,} rows in {num_chunks} chunks...",
        fg="cyan",
        bold=True,
    )

    total_processed = 0
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_rows)
        df_chunk = df.iloc[start_idx:end_idx].copy()

        num_records = process_chunk(df_chunk, chunk_idx, temp_dir)
        total_processed += num_records

        del df_chunk
        gc.collect()

    # Free memory from original dataframe
    del df
    gc.collect()

    # Concatenate all chunks
    click.secho(f"\nConcatenating {num_chunks} chunks...", fg="cyan", bold=True)
    chunk_dirs = sorted(temp_dir.glob("chunk_*"))

    datasets = []
    for chunk_dir in chunk_dirs:
        ds = load_from_disk(str(chunk_dir))
        datasets.append(ds)
        click.secho(f"  Loaded {chunk_dir.name}: {len(ds):,} records", fg="white")

    final_dataset = concatenate_datasets(datasets)
    click.secho(f"Concatenated dataset: {len(final_dataset):,} records", fg="green")

    # Filter by minimum prefix length
    if min_prefix_len > 0:
        click.secho(f"\nFiltering prefixes >= {min_prefix_len} chars...", fg="cyan")
        before_filter = len(final_dataset)
        final_dataset = final_dataset.filter(
            lambda x: len(x.get("prefix", "") or "") >= min_prefix_len,
            num_proc=4,
            desc="Filtering prefix length",
        )
        click.secho(
            f"  -> Kept {len(final_dataset):,} / {before_filter:,} ({100*len(final_dataset)/before_filter:.1f}%)",
            fg="green",
        )

    # Undersample if requested
    if undersample and len(final_dataset) > undersample:
        click.secho(f"\nUndersampling to {undersample:,} records...", fg="cyan")
        final_dataset = final_dataset.shuffle(seed=42).select(range(undersample))
        click.secho(f"  -> Final size: {len(final_dataset):,} records", fg="green")

    # Final statistics
    click.secho(f"\nFinal Statistics:", fg="cyan", bold=True)
    click.secho(f"  Total records: {len(final_dataset):,}", fg="white")
    click.secho(f"  Columns: {list(final_dataset.column_names)}", fg="white")

    # Show examples
    click.secho(f"\nExamples with historical searches:", fg="cyan", bold=True)
    shown = 0
    for i in range(min(1000, len(final_dataset))):
        if shown >= 3:
            break
        example = final_dataset[i]
        hist = example.get("historical_searches", [])
        if len(hist) > 0:
            shown += 1
            click.secho(f"\n  Example {shown}:", fg="white", bold=True)
            click.secho(f"    prefix: {example.get('prefix', 'N/A')}", fg="green")
            click.secho(
                f"    final_search_term: {example.get('final_search_term', 'N/A')}",
                fg="green",
            )
            click.secho(
                f"    historical_searches ({len(hist)}): {hist[:5]}{'...' if len(hist) > 5 else ''}",
                fg="green",
            )

    # Save final dataset
    click.secho(f"\nSaving final dataset to: {output_path}", fg="cyan")
    final_dataset.save_to_disk(output_path)
    click.secho("Saved!", fg="green")

    # Cleanup temp chunks
    click.secho("\nCleaning up temporary chunks...", fg="cyan")
    import shutil

    shutil.rmtree(temp_dir)
    click.secho("Done!", fg="green")

    click.secho("\n" + "=" * 60, fg="bright_yellow", bold=True)
    click.secho("COMPLETE", fg="bright_green", bold=True)
    click.secho("=" * 60 + "\n", fg="bright_yellow", bold=True)

    return final_dataset


if __name__ == "__main__":
    main()
