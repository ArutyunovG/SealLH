from seallh.helpers.transform.transform import Transform

import albumentations as A

from typing import Dict

class AlbumentationsTransform(Transform):

    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, data: Dict) -> Dict:
        return self.transform(**data)
