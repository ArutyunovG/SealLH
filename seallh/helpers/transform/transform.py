from abc import ABC, abstractmethod
from typing import Dict

class Transform(ABC):

    def __init__(self, cfg = None):
        if cfg is not None:
            for key, value in cfg.items():
                setattr(self, key, value)

    @abstractmethod
    def __call__(self, data: Dict) -> Dict:
        pass
