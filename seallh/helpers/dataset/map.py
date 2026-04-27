from torch.utils.data import Dataset

from typing import Callable

class Map(Dataset):

    def __init__(self, 
                 dataset: Dataset,
                 func: Callable):

        assert isinstance(dataset, Dataset)
        assert callable(func)

        self.dataset = dataset
        self.func = func


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx: int):
        return self.func(self.dataset[idx])


    def __getattr__(self, name):
        return getattr(self.dataset, name)
