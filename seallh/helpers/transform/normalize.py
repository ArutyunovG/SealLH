from seallh.helpers.transform.albumentations_transform import AlbumentationsTransform

from albumentations import Normalize as _Normalize

class Normalize(AlbumentationsTransform):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.transform = _Normalize(**cfg)
