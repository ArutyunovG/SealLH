from typing import List, Dict, Any

from torch.utils.data import Dataset

from seallh.helpers.dataset.modifiers.base_modifier import DatasetModifier
from seallh.experiment.utils import import_class


class ComposeModifier(DatasetModifier):
    """Apply a sequence of modifiers in order.

    Args:
        modifiers: List of modifier configs, each with 'class' and optional 'args'.
    """

    def __init__(self, modifiers: List[Dict[str, Any]]):
        self.modifiers = []
        for cfg in modifiers:
            cls = import_class(cfg["class"])
            assert issubclass(cls, DatasetModifier), f"{cls} is not a DatasetModifier"
            args = cfg.get("args", {})
            self.modifiers.append(cls(**args))

    def __call__(self, dataset: Dataset) -> Dataset:
        for modifier in self.modifiers:
            dataset = modifier(dataset)
        return dataset
