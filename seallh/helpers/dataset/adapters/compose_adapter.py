from typing import List, Dict, Any
from seallh.helpers.dataset.adapters.base_adapter import BaseAdapter
from seallh.experiment.utils import import_class


class ComposeAdapter(BaseAdapter):

    def __init__(self, dataset, adapters: List[Dict[str, Any]]):
        super().__init__(dataset)
        self._adapters = []
        for adapter_cfg in adapters:
            cls = import_class(adapter_cfg["class"])
            args = adapter_cfg.get("args", {})
            ds = cls(dataset, **args)
            self._adapters.append(ds)

    def __call__(self, data):
        for adapter in self._adapters:
            data = adapter(data)
        return data

