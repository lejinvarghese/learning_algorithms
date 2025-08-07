from datasets import load_dataset, DatasetDict, Dataset
from click import secho
import numpy as np

class ProductSearchDeduplicator:
    def __init__(self, num_partitions: int = 10):
        self.num_partitions = num_partitions
        
    def deduplicate_and_publish(self, repo_id: str = "lv12/ProductSearchDataset", subset_name: str = "triplets"):
        # Load dataset with parallel processing
        datasets = {}
        for split in ["train", "test"]:
            dataset = load_dataset(repo_id, name=subset_name, num_proc=14, split=split, ignore_verifications=True).to_pandas()
            datasets[split] = dataset
        
        # Deduplicate each split
        dedup_splits = {}
        for split in datasets:
            df = datasets[split]
            # Include negative in deduplication columns
            dedup_df = df.drop_duplicates(subset=['anchor', 'positive', 'source', 'negative'], keep='first')
            secho(f"{split}: Removed {len(df) - len(dedup_df)} duplicates", fg="blue")
            
            # Split into partitions
            partitions = np.array_split(dedup_df, self.num_partitions)
            for i, partition in enumerate(partitions):
                name = f"{split}_dedup_{i}"
                dedup_splits[name] = Dataset.from_pandas(partition, preserve_index=False)
        
        # Push to hub with original subset name
        secho(f"Pushing deduplicated dataset to {repo_id}", fg="yellow")
        DatasetDict(dedup_splits).push_to_hub(
            repo_id,
            private=False,
            config_name=subset_name
        )
        secho("Successfully pushed deduplicated dataset", fg="green")

if __name__ == "__main__":
    deduplicator = ProductSearchDeduplicator(num_partitions=10)
    deduplicator.deduplicate_and_publish() 