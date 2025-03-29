from abc import ABC, abstractmethod
from datasets import load_dataset
from multiprocessing import cpu_count

N_PROC = cpu_count() - 1
RANDOM_STATE = 42


class Dataset(ABC):
    def __init__(self, repo_id, sample_size=None):
        self._repo_id = repo_id
        self._sample_size = sample_size

    @property
    def repo_id(self):
        return self._repo_id

    @abstractmethod
    def convert_triplets(self):
        pass

    @abstractmethod
    def create_query(self):
        pass

    @abstractmethod
    def create_document(self):
        pass

    def load(self, split="train"):
        data = load_dataset(self.repo_id, num_proc=N_PROC, split=split)
        if self._sample_size is None:
            return data
        else:
            return data.shuffle(seed=RANDOM_STATE).select(range(self._sample_size))

    def get_negatives(self):
        pass


class HomeDepotDataset(Dataset):
    def __init__(self, repo_id="bstds/home_depot", sample_size=None):
        super().__init__(repo_id, sample_size)

    def convert_triplets(self):
        pass

    def create_query(self):
        pass

    def create_document(self):
        pass
