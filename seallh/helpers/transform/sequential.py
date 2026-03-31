from seallh.helpers.transform import Transform
from seallh.experiment.utils import import_class

from typing import Dict, List, Sequence

from omegaconf import DictConfig


class Sequential(Transform):

    def __init__(self, transforms: Sequence):

        self.transforms: List[Transform] = []

        for transform in transforms:
            if isinstance(transform, Transform):
                self.transforms.append(Transform)
            else:
                assert isinstance(transform, DictConfig)
                transform_cls = import_class(transform["class"])
                self.transforms.append(transform_cls(transform.args))


    def __call__(self, data: Dict) -> Dict:

        assert isinstance(data, Dict)

        for transform in self.transforms:
            data = transform(data)
            assert isinstance(data, Dict)
        
        return data
