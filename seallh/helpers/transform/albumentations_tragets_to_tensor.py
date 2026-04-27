from seallh.helpers.transform.transform import Transform

import torch

from typing import Dict

class AlbumentationTargetsToTensor(Transform):

    def __call__(self, data: Dict) -> Dict:
        if "bboxes" in data:
            data["bboxes"] = torch.tensor(data["bboxes"], dtype=torch.float32).reshape(-1, 4)
        if "labels" in data:
            data["labels"] = torch.tensor(data["labels"], dtype=torch.long)
        return data
