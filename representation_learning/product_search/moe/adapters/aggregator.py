from typing import Optional
from click import secho
from datasets import Dataset, DatasetDict, concatenate_datasets
from adapters import AmazonDataset, BaseDataset, CrowdFlowerDataset, GoogleDataset, HomeDepotDataset, WayfairDataset
import pandas as pd

DATASET_NAME = "lv12/ProductSearchDataset"


class DatasetAggregator:
    def __init__(
        self,
        sample_size: Optional[int] = None,
        chunk_size: Optional[int] = None,
        splits: list[str] = ["train", "test"],
    ):
        self.sources = [AmazonDataset, WayfairDataset, HomeDepotDataset, CrowdFlowerDataset]
        self.sources = [GoogleDataset]
        self.sample_size = sample_size
        self.chunk_size = chunk_size
        self.splits = splits
        self.datasets = self.generate_datasets()
        self.subsets = {}

    def generate_datasets(self) -> dict[str, list[BaseDataset]]:
        """Generate datasets."""
        datasets = {}
        for split in self.splits:
            dataset_splits = []
            for source in self.sources:
                try:
                    dataset = source(sample_size=self.sample_size, chunk_size=self.chunk_size, split=split)
                    dataset_splits.append(dataset)
                except Exception as e:
                    secho(f"Error loading dataset: {e}", fg="red")
                    continue
            datasets[split] = dataset_splits
        return datasets

    def identify_max_chunks(self) -> int:
        """Generate the maximum number of chunks for triplets."""
        max_chunks = 1
        for split in self.splits:
            dataset_splits = self.datasets.get(split, [])
            for dataset in dataset_splits:
                max_chunks = max(max_chunks, dataset._max_chunks)
        secho(f"Max chunks: {max_chunks}", fg="green")
        return max_chunks

    def generate_pairs(self) -> Dataset:
        """Generate pairs from all datasets and concatenate them."""
        splits = {}
        for split in self.splits:
            dataset_splits = self.datasets.get(split, [])
            if len(dataset_splits) > 0:
                pairs_list = []
                for dataset in dataset_splits:
                    pairs = dataset.generate_pairs()
                    pairs_list.append(pairs)

                combined_pairs = concatenate_datasets(pairs_list)
                secho(f"Total pairs: {len(combined_pairs)}", fg="blue")
                splits[split] = combined_pairs
        self.subsets["pairs"] = splits
        return splits

    def generate_triplets(self, chunk_index: int) -> Dataset:
        """Generate triplets from all datasets and concatenate them."""
        splits = {}
        for split in self.splits:
            dataset_splits = self.datasets.get(split, [])
            if len(dataset_splits) > 0:
                triplets_list = []
                for dataset in dataset_splits:
                    # Generate triplets for this chunk
                    triplets = dataset.generate_triplets(chunk_index=chunk_index)
                    if triplets is not None:  # Only append if we got valid triplets
                        triplets_list.append(triplets)
                
                if triplets_list:
                    # Combine all triplets for this split
                    combined_triplets = concatenate_datasets(triplets_list)
                    secho(f"Total triplets for {split}: {len(combined_triplets)}", fg="blue")
                    splits[split] = combined_triplets
                
        self.subsets["triplets"] = splits
        return splits

    def push_to_hub(
        self,
        repo_id: str = DATASET_NAME,
        private: bool = False,
        subset_name: str = "pairs",
        chunk_index: Optional[int] = None,
        chunk_suffix: Optional[str] = None,
    ) -> None:
        """Push the dataset to HuggingFace Hub."""
        secho(f"Pushing the dataset to {repo_id}", fg=(229, 192, 123))

        if subset_name not in self.subsets:
            raise ValueError(f"Subset {subset_name} not found in the dataset.")
        subset = self.subsets[subset_name]

        # Create a new dictionary instead of modifying the existing one
        if chunk_index is not None:
            new_subset = {}
            for key, value in subset.items():
                name = f"{key}_{chunk_index}"
                if chunk_suffix:
                    name = f"{name}_{chunk_suffix}_skip_1"
                new_subset[name] = value
            subset = new_subset

        # Push to hub
        DatasetDict(subset).push_to_hub(
            repo_id,
            private=private,
            config_name=subset_name,
        )

        secho(f"Successfully pushed the dataset to {repo_id}", fg="green")
        
        # Clear memory after pushing
        del subset
        import gc
        gc.collect()
