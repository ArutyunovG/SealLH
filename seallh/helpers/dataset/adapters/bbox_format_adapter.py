from typing import List
from torch.utils.data import Dataset

from seallh.helpers.dataset.adapters.base_adapter import BaseAdapter


class BBoxFormatAdapter(BaseAdapter):

    def __init__(self, dataset: Dataset, source_format: str, target_format: str):
        super().__init__(dataset)
        method_name = f"{source_format}_to_{target_format}"
        if not hasattr(self, method_name):
            raise ValueError(
                f"Unsupported conversion: {source_format} -> {target_format}. "
                f"No method '{method_name}' found."
            )
        self.convert = getattr(self, method_name)

    def __call__(self, data):
        bboxes_key = self.resolve_bbox_key(data)
        data[bboxes_key] = self.convert(data[bboxes_key])
        return data

    @staticmethod
    def xywh_to_xyxy(bboxes: List[List[float]]) -> List[List[float]]:
        return [[x, y, x + w, y + h] for x, y, w, h in bboxes]

    @staticmethod
    def xyxy_to_xywh(bboxes: List[List[float]]) -> List[List[float]]:
        return [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in bboxes]
