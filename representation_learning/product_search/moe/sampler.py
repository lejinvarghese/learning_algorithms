from datasets import load_dataset, Dataset
from click import secho

def sample_dataset(
    repo_id: str = "lv12/ProductSearchDataset",
    subset_name: str = "triplets",
    sample_size: int = 1000,
    output_path: str = "samples"
):
    """Sample from a Hugging Face dataset and save to disk."""
    # Load dataset
    dataset = load_dataset(repo_id, subset_name, num_proc=14, ignore_verifications=True)
    
    # Sample from each split
    for split in dataset:
        # Sample directly from the dataset
        sample = dataset[split].shuffle(seed=42).select(range(min(sample_size, len(dataset[split]))))
        
        # Save sample
        output_file = f"{output_path}/{subset_name}_{split}_sample"
        sample.save_to_disk(output_file)
        secho(f"Saved {len(sample)} examples from {split} to {output_file}", fg="green")
        
        # Print sample statistics
        secho(f"\nSample statistics for {split}:", fg="blue")
        secho(f"Total rows: {len(sample)}", fg="blue")
        secho(f"Features: {', '.join(sample.features.keys())}", fg="blue")
        secho(f"Size: {sample.info.size_in_bytes / 1024 / 1024:.2f} MB", fg="blue")

if __name__ == "__main__":
    sample_dataset(
        repo_id="lv12/ProductSearchDataset",
        subset_name="triplets",
        sample_size=1000,
        output_path="samples"
    ) 