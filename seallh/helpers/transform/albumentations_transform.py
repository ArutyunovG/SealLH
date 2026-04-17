from seallh.helpers.transform.transform import Transform

import albumentations as A
import numpy as np

from typing import Dict

class AlbumentationsTransform(Transform):

    def __init__(self, cls, **cfg):
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
        has_bboxes = (
            'bboxes' in data
            and 'bboxes' in self.transform.targets
        )

        if has_bboxes:
            data = dict(data)
            bboxes_list = data['bboxes']
            if len(bboxes_list):
                h, w = image.shape[:2]
                bboxes = np.array(bboxes_list, dtype=np.float32)[:, :4]
                bboxes[:, [0, 2]] /= w
                bboxes[:, [1, 3]] /= h
                # Append an index column so we can track which bboxes survive
                # transforms that call validate_bboxes (e.g. Rotate).
                indices = np.arange(len(bboxes), dtype=np.float32).reshape(-1, 1)
                data['bboxes'] = np.hstack([bboxes, indices])
            else:
                data['bboxes'] = np.zeros((0, 5), dtype=np.float32)

        result = self.transform(**data)

        if has_bboxes and 'bboxes' in result:
            out = np.array(result['bboxes'], dtype=np.float32)
            if len(out):
                surviving = out[:, 4].astype(int)
                out_bboxes = out[:, :4].copy()
                h, w = result['image'].shape[:2]
                out_bboxes[:, [0, 2]] *= w
                out_bboxes[:, [1, 3]] *= h
                result['bboxes'] = out_bboxes
                # Keep labels in sync with surviving bboxes
                if 'labels' in result and len(surviving) < len(data.get('bboxes', [])):
                    labels = result['labels']
                    if isinstance(labels, np.ndarray):
                        result['labels'] = labels[surviving]
                    elif isinstance(labels, (list, tuple)):
                        result['labels'] = [labels[i] for i in surviving]
            else:
                result['bboxes'] = np.zeros((0, 4), dtype=np.float32)
                if 'labels' in result:
                    labels = result['labels']
                    if isinstance(labels, np.ndarray):
                        result['labels'] = labels[:0]
                    elif isinstance(labels, (list, tuple)):
                        result['labels'] = []

        return result
    
    def __repr__(self) -> str:
        return f"AlbumentationsTransform({self.transform!r})"


from albumentations import Normalize as _Normalize
from albumentations import RandomBrightnessContrast as _RandomBrightnessContrast
from albumentations import Resize as _Resize
from albumentations import Rotate as _Rotate
from albumentations import HorizontalFlip as _HorizontalFlip
from albumentations import VerticalFlip as _VerticalFlip

from albumentations.pytorch import ToTensorV2 as _ToTensor

HorizontalFlip = lambda **cfg: AlbumentationsTransform(_HorizontalFlip, **cfg)
Normalize = lambda **cfg: AlbumentationsTransform(_Normalize, **cfg)
RandomBrightnessContrast = lambda **cfg: AlbumentationsTransform(_RandomBrightnessContrast, **cfg)
Resize = lambda **cfg: AlbumentationsTransform(_Resize, **cfg)
Rotate = lambda **cfg: AlbumentationsTransform(_Rotate, **cfg)
VerticalFlip = lambda **cfg: AlbumentationsTransform(_VerticalFlip, **cfg)

ToTensor = lambda **cfg: AlbumentationsTransform(_ToTensor, **cfg)
