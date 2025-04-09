from click import secho
from datasets import Dataset, load_dataset
import random
from multiprocessing import cpu_count

from adapters.core import RANDOM_STATE, BaseDataset
from adapters.miners import HardNegativeMiner

FEATURE_COLUMNS = [
    "query",
    "product_id",
    "title",
    "score_reciprocal",
]


class GoogleDataset(BaseDataset):
    def __init__(
        self,
        repo_id="Marqo/marqo-GS-10M",
        sample_size=None,
        chunk_size=1000,
        split="train",
        cols=FEATURE_COLUMNS,
    ):
        self._repo_id = repo_id
        self._sample_size = sample_size
        self._chunk_size = chunk_size
        self._num_procs = cpu_count() - 1
        self._split = split
        self.name = "google"
        
        actual_split = "in_domain" if split == "train" else "zero_shot" if split == "test" else split
        
        if self._sample_size:
            try:
                query_data = load_dataset(self._repo_id, split=actual_split, columns=["query"], num_proc=self._num_procs)
                unique_queries = list(set(query_data["query"]))
                if self._sample_size < len(unique_queries):
                    secho(f"Sampling {self._sample_size} queries from {len(unique_queries)} total queries", fg="green")
                    queries_to_load = set(random.sample(unique_queries, self._sample_size))
                else:
                    self._sample_size = None
            except Exception as e:
                secho(f"Warning: Could not efficiently load queries for sampling: {e}. Loading full data.", fg="yellow")
                queries_to_load = None

        self._data = self.load(split, cols, queries_to_load)

        self.generate_query(queries_already_sampled=(queries_to_load is not None or self._sample_size is None))
        self.generate_document()
        self._map_relevance()

    def _map_relevance(self):
        if not hasattr(self, '_data') or not self._data:
            secho("Warning: _data not available for _map_relevance.", fg="yellow")
            return
        if "score_reciprocal" not in self._data.column_names:
            secho("Warning: 'score_reciprocal' column missing for _map_relevance.", fg="yellow")
            return
            
        self._data = self._data.map(
            lambda x: {"relevance": round(1 + (x.get("score_reciprocal", 0.0) / 100) * 2, 2)},
            num_proc=self._num_procs,
            remove_columns=["score_reciprocal"],
        )

    def load(self, split: str, cols: list[str] = FEATURE_COLUMNS, queries_to_load: set = None):
        secho(
            f"Loading data from {self._repo_id} (split: {split})...",
            fg=(229, 192, 123),
        )
        if split == "train":
            actual_split = "in_domain"
        elif split == "test":
            actual_split = "zero_shot"
        else:
            actual_split = split
            
        try:
             data = load_dataset(self.repo_id, split=actual_split, columns=cols, num_proc=self._num_procs)
        except Exception as e:
             secho(f"ERROR loading dataset {self.repo_id} split {actual_split}: {e}", fg="red")
             schema_dict = {col: [] for col in cols} if cols else {}
             return Dataset.from_dict(schema_dict) 

        if queries_to_load:
             if "query" in data.column_names:
                  original_count = len(data)
                  data = data.filter(lambda x: x["query"] in queries_to_load, num_proc=self._num_procs)
             else:
                  secho("Warning: 'query' column not found, cannot filter by pre-sampled queries.", fg="yellow")
        elif self._sample_size is not None:
             secho("Applying sampling after full load (fallback).", fg="yellow")
             if "query" in data.column_names:
                 unique_queries = list(set(data.unique("query")))
                 if self._sample_size < len(unique_queries):
                      selected_queries = random.sample(unique_queries, self._sample_size)
                      data = data.filter(lambda x: x["query"] in selected_queries, num_proc=self._num_procs)
             else:
                  secho("Warning: 'query' column not found, cannot apply fallback sampling.", fg="yellow")

        secho(f"Finished loading/filtering for {self.name}. Final record count: {len(data)}", fg="green")
        
        return data.shuffle(seed=RANDOM_STATE)

    def generate_document(self):
        if not hasattr(self, '_data') or not self._data:
             secho("Warning: _data not available for generate_document.", fg="yellow")
             return
             
        cols_to_remove = ["product_id", "title"]
        available_cols_to_remove = [col for col in cols_to_remove if col in self._data.column_names]
        
        self._data = self._data.map(
            lambda row: {"document": self.format_document(title=row.get("title"))},
            remove_columns=available_cols_to_remove, 
            num_proc=self._num_procs,
        )
        if "document" in self._data.column_names:
             self._n_documents = len(set(self._data.unique("document")))
        else:
             self._n_documents = 0
             secho("Warning: 'document' column not created in generate_document.", fg="yellow")

    def generate_triplets(self, threshold=1.0, chunk_index: int = None):
        return super().generate_triplets(threshold=threshold, chunk_index=chunk_index)

    def generate_negatives(self, threshold=0.8, data_subset=None):
        data_to_process = data_subset if data_subset is not None else self._data
        
        if not data_to_process or len(data_to_process) == 0:
             return Dataset.from_dict({"anchor": [], "negative": [], "relevance": []})

        required_miner_cols = ["query", "document"]
        if not all(col in data_to_process.column_names for col in required_miner_cols):
             secho(f"ERROR: Missing required columns {required_miner_cols} for HardNegativeMiner input.", fg="red")
             return Dataset.from_dict({"anchor": [], "negative": [], "relevance": []})

        miner_input = data_to_process.map(
            lambda x: {"anchor": x["query"], "document": x["document"]},
            num_proc=self._num_procs,
            remove_columns=[c for c in data_to_process.column_names if c not in required_miner_cols]
        )

        try:
            miner = HardNegativeMiner(dataset=miner_input, max_score=threshold)
            neg = miner.run()
            expected_cols = ["anchor", "negative", "relevance"]
            if not all(col in neg.column_names for col in expected_cols):
                 secho(f"ERROR: HardNegativeMiner output schema mismatch. Expected {expected_cols}, got {neg.column_names}.", fg="red")
                 return Dataset.from_dict({col: [] for col in expected_cols})
        except Exception as e:
            secho(f"ERROR running HardNegativeMiner: {e}", fg="red")
            return Dataset.from_dict({"anchor": [], "negative": [], "relevance": []})

        return neg
