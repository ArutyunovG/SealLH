from typing import Dict, List, Optional
from torch.utils.data import Dataset

from seallh.helpers.dataset.adapters.base_adapter import BaseAdapter


class CategoryMappingAdapter(BaseAdapter):

    def __init__(
        self,
        dataset: Dataset,
        mapping: Dict[int, int],
        names: Optional[Dict[int, str]] = None,
    ):
        """
        Args:
            dataset: wrapped dataset
            mapping: {source_label: target_label}, e.g. {0: 0, 1: 0, 2: 1}
            names: {target_label: name}, e.g. {0: "person", 1: "vehicle"}.
                   If None, names are not remapped.
                   If provided, must cover every target label in mapping values.
        """
        super().__init__(dataset)
        self.mapping = mapping

        if names is not None:
            missing = set(mapping.values()) - set(names.keys())
            if missing:
                raise ValueError(
                    f"names must cover all target labels. Missing: {missing}"
                )

        self.names = names

    def __call__(self, data):
        labels_key = self.resolve_label_key(data)
        bboxes_key = self.resolve_bbox_key(data)

        labels = data[labels_key]
        bboxes = data.get(bboxes_key, [])

        new_labels = []
        new_bboxes = []
        for i, label in enumerate(labels):
            if label not in self.mapping:
                continue
            new_labels.append(self.mapping[label])
            if bboxes:
                new_bboxes.append(bboxes[i])

        data[labels_key] = new_labels
        if bboxes:
            data[bboxes_key] = new_bboxes

        return data

    @property
    def categories(self) -> List[str]:
        if self.names is not None:
            max_id = max(self.names.keys())
            return [self.names[i] for i in range(max_id + 1)]
        return self.dataset.categories

    def __getattr__(self, name):
        return getattr(self.dataset, name)
