from seallh.helpers.transform.albumentations_transform import AlbumentationsTransform

from albumentations import Resize as _Resize

class Resize(AlbumentationsTransform):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.transform = _Resize(**cfg)
