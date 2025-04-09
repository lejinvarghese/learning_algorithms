import json
import re
from abc import ABC
from multiprocessing import cpu_count

from click import secho
import random
            
from datasets import Dataset, load_dataset, Features, Value

RANDOM_STATE = 42
random.seed(RANDOM_STATE)

DATASET_CHUNK_SIZES = { # Define specific chunk sizes here
    "wayfair": 100,
    "amazon": 5000,
    # Add other datasets if needed
}
DEFAULT_CHUNK_SIZE = 1000

class BaseDataset(ABC):
    def __init__(
        self,
        repo_id: str,
        sample_size: int = None,
        chunk_size: int = 1000,
        split="train",
        cols: list[str] = None,
    ):
        self._repo_id = repo_id
        self._sample_size = sample_size
        self._chunk_size = chunk_size
        self._num_procs = cpu_count() - 1
        self._split = split
        self._data = self.load(split, cols)
        secho(f"Total records loaded: {len(self._data)}", fg="green")

    @property
    def repo_id(self):
        return self._repo_id

    @property
    def data(self):
        return self._data

    @property
    def n_queries(self):
        return self._n_queries

    @property
    def n_documents(self):
        return self._n_documents

    def generate_query(self, queries_already_sampled=False):
        secho(f"Generating queries for {self.name} dataset...", fg="blue")
        
        self._unique_queries = list(set(self._data.unique("query")))
        self._n_queries = len(self._unique_queries)

        if not queries_already_sampled and self._sample_size is not None and self._sample_size < self._n_queries:
             secho(f"Applying sampling in BaseDataset.generate_query: {self._sample_size} queries", fg="yellow")
             sampled_queries = random.sample(self._unique_queries, self._sample_size)
             self._unique_queries = sampled_queries
             self._n_queries = len(self._unique_queries)
             self._data = self._data.filter(
                 lambda x: x["query"] in self._unique_queries,
                 num_proc=self._num_procs
             )

        chunks = {}
        effective_chunk_size = DATASET_CHUNK_SIZES.get(getattr(self, 'name', None), self._chunk_size or DEFAULT_CHUNK_SIZE)

        if self._n_queries > 0 and effective_chunk_size > 0:
            for i in range(0, self._n_queries, effective_chunk_size):
                chunk_index = i // effective_chunk_size
                chunks[chunk_index] = self._unique_queries[i:i + effective_chunk_size]
        else:
            chunks = {0: self._unique_queries}

        self._max_chunks = len(chunks)
        self._query_chunks = chunks
        secho(f"Total query chunks created: {self._max_chunks}", fg="blue")

    def generate_document(self):
        pass

    @staticmethod
    def strip_html(text):
        if not isinstance(text, str):
            return ""
        clean = re.compile("<.*?>")
        return re.sub(clean, "", text)

    @staticmethod
    def format_document(**kwargs):
        if kwargs.get("title"):
            template = f"""**product title**: {kwargs.get('title')}\n"""
        else:
            template = """"""
        if kwargs.get("category"):
            template += f"""**product category**: {kwargs.get('category').replace(" / ", " > ")}\n"""
        if kwargs.get("attributes"):
            template += """**product attributes**:\n"""
            for k, v in kwargs.get("attributes").items():
                template += f""" - **{k}**: {v}\n"""

        if kwargs.get("description"):
            template += f"""**product description**: {kwargs.get('description')}"""
        return BaseDataset.strip_html(template.strip().lower())

    def load(self, split: str, cols: list[str] = None):
        secho(
            f"Loading data from {self._repo_id} for split {split} using: {self._num_procs} cores",
            fg=(229, 192, 123),
        )
        data = load_dataset(self._repo_id, num_proc=self._num_procs, split=split, columns=cols)
        secho(f"Total records loaded: {len(data)}", fg="green")
        return data

    def generate_pairs(self):
        pairs = self._data
        source = [self.name] * len(pairs)
        pairs = pairs.add_column("source", source)
        secho(f"Generated {len(pairs)} pairs.", fg="green")
        secho(f"Queries: {self.n_queries}, Documents: {self.n_documents}.", fg="green")
        return pairs

    def generate_triplets(self, threshold=3.0, chunk_index: int = None):
        if chunk_index is not None:
             secho(f"Generating triplets for {self.name} chunk {chunk_index}...", fg="blue")
        
        chunk_data = self._data
        
        if chunk_index is not None and self._query_chunks and chunk_index in self._query_chunks:
            chunk_queries = set(self._query_chunks[chunk_index])
            if not chunk_queries:
                 return self._create_empty_triplet_dataset()
                 
            chunk_data = self._data.filter(
                lambda x: x["query"] in chunk_queries,
                num_proc=self._num_procs
            )
        elif chunk_index is not None:
             secho(f"Warning: Chunk index {chunk_index} not found or query chunks empty.", fg="yellow")
             return self._create_empty_triplet_dataset()
        
        positives_ds = self.generate_positives(threshold=threshold, data_subset=chunk_data)
        negatives_ds = self.generate_negatives(threshold=threshold, data_subset=chunk_data)

        if len(positives_ds) == 0 or len(negatives_ds) == 0:
            return self._create_empty_triplet_dataset()

        try:
             positives = positives_ds.to_pandas()
             negatives = negatives_ds.to_pandas()
        except Exception as e:
             secho(f"Error converting dataset subset to pandas (chunk {chunk_index}): {e}", fg="red")
             return self._create_empty_triplet_dataset()

        positives = positives.rename(columns={"positive": "document", "relevance": "relevance_positive"})
        negatives = negatives.rename(columns={"negative": "document", "relevance": "relevance_negative"})

        if "anchor" not in positives.columns or "anchor" not in negatives.columns:
             secho("Error: 'anchor' column missing before merge.", fg="red")
             return self._create_empty_triplet_dataset()
        if "relevance_positive" not in positives.columns:
             positives['relevance_positive'] = threshold
             secho("Warning: 'relevance' column missing in positives, added default.", fg="yellow")
        if "relevance_negative" not in negatives.columns:
             negatives['relevance_negative'] = threshold - 0.1
             secho("Warning: 'relevance' column missing in negatives, added default.", fg="yellow")

        try:
            triplets = positives.merge(negatives, on="anchor", suffixes=("_pos", "_neg")) 
        except Exception as e:
             secho(f"Error merging pandas DataFrames (chunk {chunk_index}): {e}", fg="red")
             return self._create_empty_triplet_dataset()

        if triplets.empty:
             return self._create_empty_triplet_dataset()
             
        triplets["margin"] = round(triplets["relevance_positive"] - triplets["relevance_negative"], 2)
        triplets["source"] = self.name
        triplets = triplets.rename(columns={"document_pos": "positive", "document_neg": "negative"})

        metadata_cols = [col for col in ['relevance_positive', 'relevance_negative'] if col in triplets.columns]
        if metadata_cols:
             try:
                 triplets["metadata"] = triplets[metadata_cols].apply(lambda x: json.dumps(x.to_dict()), axis=1)
                 triplets = triplets.drop(columns=metadata_cols)
             except Exception as e:
                  secho(f"Error creating metadata JSON (chunk {chunk_index}): {e}", fg="yellow")
                  triplets["metadata"] = "{}"
        else:
             triplets["metadata"] = "{}"

        final_cols = ["anchor", "positive", "negative", "margin", "source", "metadata"]
        missing_cols = [col for col in final_cols if col not in triplets.columns]
        if missing_cols:
             secho(f"Error: Final columns missing before Dataset creation: {missing_cols}", fg="red")
             return self._create_empty_triplet_dataset()
        
        triplets_final_df = triplets[final_cols]

        try:
            triplets_dataset = Dataset.from_pandas(triplets_final_df, preserve_index=False, features=self._get_triplet_features())
            secho(f"Generated {len(triplets_dataset)} triplets for chunk {chunk_index}.", fg="green")
            return triplets_dataset
        except Exception as e:
            secho(f"Error converting final DataFrame to Dataset (chunk {chunk_index}): {e}", fg="red")
            return self._create_empty_triplet_dataset()

    def _get_triplet_features(self):
        return Features({
            "anchor": Value("string"),
            "positive": Value("string"),
            "negative": Value("string"),
            "margin": Value("float64"),
            "source": Value("string"),
            "metadata": Value("string")
        })

    def _create_empty_triplet_dataset(self):
        return Dataset.from_dict({
            "anchor": [], "positive": [], "negative": [], 
            "margin": [], "source": [], "metadata": []
        }, features=self._get_triplet_features())

    def generate_positives(self, threshold, data_subset=None):
        data_to_process = data_subset if data_subset is not None else self._data
        if not data_to_process or len(data_to_process) == 0:
             return Dataset.from_dict({"anchor": [], "positive": [], "relevance": []})

        if "relevance" not in data_to_process.column_names:
             secho("Error: 'relevance' column missing for generate_positives.", fg="red")
             return Dataset.from_dict({"anchor": [], "positive": [], "relevance": []})
             
        pos = data_to_process.filter(lambda x: x["relevance"] >= threshold, num_proc=self._num_procs).map(
            lambda x: {"anchor": x["query"], "positive": x["document"], "relevance": x["relevance"]},
            num_proc=self._num_procs,
            remove_columns=[col for col in data_to_process.column_names if col not in ["query", "document", "relevance"]],
        )
        return pos

    def generate_negatives(self, threshold, data_subset=None):
        data_to_process = data_subset if data_subset is not None else self._data
        if not data_to_process or len(data_to_process) == 0:
             return Dataset.from_dict({"anchor": [], "negative": [], "relevance": []})
             
        if "relevance" not in data_to_process.column_names:
             secho("Error: 'relevance' column missing for generate_negatives (base).", fg="red")
             return Dataset.from_dict({"anchor": [], "negative": [], "relevance": []})

        neg = data_to_process.filter(lambda x: x["relevance"] < threshold, num_proc=self._num_procs).map(
            lambda x: {"anchor": x["query"], "negative": x["document"], "relevance": x["relevance"]},
            num_proc=self._num_procs,
            remove_columns=[col for col in data_to_process.column_names if col not in ["query", "document", "relevance"]],
        )
        return neg
