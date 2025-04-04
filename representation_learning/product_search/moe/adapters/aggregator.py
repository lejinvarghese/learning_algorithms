from typing import Optional
from datasets import concatenate_datasets, DatasetDict, Dataset
from click import secho

from adapters import BaseDataset, AmazonDataset, HomeDepotDataset, GoogleDataset, WayfairDataset

DATASET_NAME = "lv12/ProductSearchDataset"


class DatasetAggregator:
    def __init__(
        self,
        sample_size: Optional[int] = None,
        splits: list[str] = ["train", "test"],
    ):
        self.sources = [HomeDepotDataset, AmazonDataset, WayfairDataset, GoogleDataset]
        self.sources = [GoogleDataset]
        self.sample_size = sample_size
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
                    dataset = source(sample_size=self.sample_size, split=split)
                    dataset_splits.append(dataset)
                except Exception as e:
                    secho(f"Error loading dataset: {e}", fg="red")
                    continue
            datasets[split] = dataset_splits
        return datasets

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

    def generate_triplets(self) -> Dataset:
        """Generate triplets from all datasets and concatenate them."""

        splits = {}
        for split in self.splits:
            dataset_splits = self.datasets.get(split, [])
            if len(dataset_splits) > 0:
                triplets_list = []
                for dataset in dataset_splits:
                    triplets = dataset.generate_triplets()
                    triplets_list.append(triplets)

                combined_triplets = concatenate_datasets(triplets_list)
                secho(f"Total triplets: {len(combined_triplets)}", fg="blue")
                splits[split] = combined_triplets
        self.subsets["triplets"] = splits
        return splits

    def push_to_hub(
        self,
        repo_id: str = DATASET_NAME,
        private: bool = False,
    ):
        """Push the dataset to HuggingFace Hub."""
        secho(f"Pushing the dataset to {repo_id}", fg=(229, 192, 123))

        for name, subset in self.subsets.items():
            DatasetDict(subset).push_to_hub(
                repo_id,
                private=private,
                config_name=name,
            )

        secho(f"Successfully pushed the dataset to {repo_id}", fg="green")
