from abc import ABC, abstractmethod

from multiprocessing import cpu_count
from click import secho

from datasets import load_dataset

RANDOM_STATE = 42


class BaseDataset(ABC):
    def __init__(
        self,
        repo_id: str,
        sample_size: int = None,
        split="train",
        cols: list[str] = None,
    ):
        self._repo_id = repo_id
        self._sample_size = sample_size

        self._num_procs = cpu_count() - 1
        self._data = self.load(split, cols)
        secho(f"Total records loaded: {len(self._data)}", fg="green")

    @property
    def repo_id(self):
        return self._repo_id

    @property
    def data(self):
        return self._data

    @abstractmethod
    def generate_pairs(self):
        pass

    @abstractmethod
    def generate_triplets(self):
        pass

    @abstractmethod
    def generate_query(self):
        pass

    @abstractmethod
    def generate_document(self):
        pass

    @staticmethod
    def format_document(**kwargs):
        if kwargs.get("title"):
            template = f"""
            **Product Title**: {kwargs.get('title')}
            """
        else:
            template = """
            """
        if kwargs.get("category"):
            template += f"""
            **Product Category**: {kwargs.get('category')}
            """
        if kwargs.get("attributes"):
            template += """
                **Product Attributes**:
                """
            for k, v in kwargs.get("attributes").items():
                template += f"""
                **{k.title()}**: {v}
                """

        if kwargs.get("description"):
            template += f"""
            **Product Description**: {kwargs.get('description')}
            """
        return template.strip().lower()

    def load(self, split: str, cols: list[str] = None):
        secho(
            f"Loading data from {self._repo_id} using: {self._num_procs} cores",
            fg="yellow",
        )
        data = load_dataset(self.repo_id, num_proc=self._num_procs, split=split, columns=cols)
        if self._sample_size is None:
            return data
        else:
            return data.shuffle(seed=RANDOM_STATE).select(range(self._sample_size))
