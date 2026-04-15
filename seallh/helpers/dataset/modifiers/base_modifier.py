from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class DatasetModifier(ABC):

    @abstractmethod
    def __call__(self, dataset: Dataset) -> Dataset:
        ...
