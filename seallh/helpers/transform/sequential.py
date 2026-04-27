from seallh.helpers.transform import Transform
from seallh.experiment.utils import import_class

from typing import Dict, List, Sequence

from omegaconf import DictConfig


class Sequential(Transform):

    def __init__(self, transforms):

        self.transforms: List[Transform] = []

        for transform in transforms:
            if isinstance(transform, Transform):
                self.transforms.append(transform)
            else:
                assert isinstance(transform, DictConfig), \
                       f"Expected DictConfig, got {type(transform)}: {transform}"
                transform_cls = import_class(transform["class"])
                args = transform.get("args", {})
                self.transforms.append(transform_cls(**args))


    def __call__(self, data: Dict) -> Dict:

        assert isinstance(data, Dict)

        for transform in self.transforms:
            data = transform(data)
            assert isinstance(data, Dict)
        
        return data

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}("]
        for i, t in enumerate(self.transforms):
            t_repr = repr(t).replace("\n", "\n  ")
            lines.append(f"  ({i}): {t_repr}")
        lines.append(")")
        return "\n".join(lines)
