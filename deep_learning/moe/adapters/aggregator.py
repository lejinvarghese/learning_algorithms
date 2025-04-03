from typing import Optional
from datasets import concatenate_datasets, DatasetDict
from click import secho

from adapters.amazon import AmazonDataset
from adapters.home_depot import HomeDepotDataset

DATASET_NAME = "lv12/ProductSearchDataset"


class DatasetAggregator:
    def __init__(
        self,
        sample_size: Optional[int] = None,
        split: str = "train",
    ):
        self.sources = [AmazonDataset, HomeDepotDataset]
        self.sample_size = sample_size
        self.split = split
        self.datasets = self.generate_datasets()

    def generate_datasets(self):
        """Generate datasets."""
        return [
            AmazonDataset(
                sample_size=self.sample_size,
                split=self.split,
            ),
            HomeDepotDataset(
                sample_size=self.sample_size,
                split=self.split,
            ),
        ]

    def generate_pairs(self):
        """Generate pairs from all datasets and concatenate them."""
        if not self.datasets:
            raise ValueError("No datasets added to aggregator")

        pairs_list = []
        for dataset in self.datasets:
            pairs = dataset.generate_pairs()
            pairs_list.append(pairs)

        combined_pairs = concatenate_datasets(pairs_list)
        secho(f"Total combined pairs: {len(combined_pairs)}", fg="blue")
        return combined_pairs

    def generate_triplets(self):
        """Generate triplets from all datasets and concatenate them."""
        if not self.datasets:
            raise ValueError("No datasets added to aggregator")

        triplets_list = []
        for dataset in self.datasets:
            triplets = dataset.generate_triplets()
            triplets_list.append(triplets)

        combined_triplets = concatenate_datasets(triplets_list)
        secho(f"Total combined triplets: {len(combined_triplets)}", fg="blue")
        return combined_triplets

    def push_to_hub(
        self,
        repo_id: str = DATASET_NAME,
        private: bool = False,
        overwrite: bool = True,
    ):
        """Push the combined dataset to HuggingFace Hub."""
        secho(f"Pushing combined dataset to {repo_id}", fg=(229, 192, 123))

        # Generate combined pairs and triplets
        pairs = self.generate_pairs()
        triplets = self.generate_triplets()

        pairs = DatasetDict({"train": pairs})
        triplets = DatasetDict({"train": triplets})

        # Push pairs subset
        pairs.push_to_hub(
            repo_id,
            private=private,
            config_name="pairs",
        )
        pairs.push_to_hub(
            repo_id,
            private=private,
            config_name="triplets",
        )

        secho(f"Successfully pushed combined dataset to {repo_id}", fg="green")
