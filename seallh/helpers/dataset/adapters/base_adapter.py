from abc import abstractmethod
from typing import Dict, Sequence
from torch.utils.data import Dataset


class BaseAdapter(Dataset):

    def __init__(self, dataset: Dataset):
        assert isinstance(dataset, Dataset), "Wrapped object must be a PyTorch Dataset"
        self.dataset = dataset

    @abstractmethod
    def __call__(self, data):
        ...

    def __getitem__(self, idx):
        return self(self.dataset[idx])

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _resolve_key(data: Dict, candidates: Sequence[str]) -> str:
        if not isinstance(data, Dict):
            raise ValueError(f"Expected data to be Dict, got {type(data)}")
        for key in candidates:
            if key in data:
                return key
        raise KeyError(
            f"No matching key found in data. Expected one of: {candidates}"
        )

    def resolve_bbox_key(self, data: Dict) -> str:
        return self._resolve_key(data, ("bboxes", "boxes", "bounding_boxes"))

    def resolve_label_key(self, data: Dict) -> str:
        return self._resolve_key(data, ("labels", "categories"))
