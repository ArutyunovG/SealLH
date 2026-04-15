from seallh.helpers.transform.transform import Transform

import albumentations as A
import numpy as np

from typing import Dict

class AlbumentationsTransform(Transform):

    def __init__(self, cls, cfg):
        super().__init__(cfg)
        self.transform = cls(**cfg)
        assert isinstance(self.transform, A.BasicTransform), \
               f"Expected an instance of albumentations.BasicTransform, got {type(self.transform)}"

    def __call__(self, data: Dict) -> Dict:
    
        assert isinstance(data, Dict), f"Expected data to be a Dict, got {type(data)}"
        assert 'image' in data, "Expected 'image' key in data for AlbumentationsTransform"
        image = data['image']
        assert isinstance(image, np.ndarray), f"Expected 'image' to be a numpy array, got {type(image)}"

        # Albumentations apply_to_bbox expects normalized [0,1] coords.
        # Mimic what A.Compose(BboxParams) does: normalize before, denormalize after.
        needs_bbox_norm = (
            'bboxes' in data and data['bboxes']
            and 'bboxes' in self.transform.targets
        )

        if needs_bbox_norm:
            h, w = image.shape[:2]
            data = dict(data)
            data['bboxes'] = [
                [b[0] / w, b[1] / h, b[2] / w, b[3] / h]
                for b in data['bboxes']
            ]

        result = self.transform(**data)

        if needs_bbox_norm and 'bboxes' in result and result['bboxes']:
            h, w = result['image'].shape[:2]
            result['bboxes'] = [
                [b[0] * w, b[1] * h, b[2] * w, b[3] * h]
                for b in result['bboxes']
            ]

        return result

from albumentations import Resize as _Resize
from albumentations import Normalize as _Normalize
from albumentations.pytorch import ToTensorV2 as _ToTensor

Resize = lambda cfg: AlbumentationsTransform(_Resize, cfg)
Normalize = lambda cfg: AlbumentationsTransform(_Normalize, cfg)
ToTensor = lambda cfg: AlbumentationsTransform(_ToTensor, cfg)
