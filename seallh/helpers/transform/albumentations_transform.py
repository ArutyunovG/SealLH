from seallh.helpers.transform.transform import Transform

import albumentations as A

from typing import Dict

class AlbumentationsTransform(Transform):

    def __init__(self, cls, cfg):
        super().__init__(cfg)
        self.transform = cls(**cfg)
        assert isinstance(self.transform, A.BasicTransform), \
               f"Expected an instance of albumentations.BasicTransform, got {type(self.transform)}"

    def __call__(self, data: Dict) -> Dict:
        assert isinstance(data, Dict)
        return self.transform(**data)

from albumentations import Resize as _Resize
from albumentations import Normalize as _Normalize
from albumentations.pytorch import ToTensorV2 as _ToTensor

Resize = lambda cfg: AlbumentationsTransform(_Resize, cfg)
Normalize = lambda cfg: AlbumentationsTransform(_Normalize, cfg)
ToTensor = lambda cfg: AlbumentationsTransform(_ToTensor, cfg)
