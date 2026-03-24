from abc import ABC, abstractmethod
from typing import Dict

class Transform(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def __call__(self, data: Dict) -> Dict:
        pass
