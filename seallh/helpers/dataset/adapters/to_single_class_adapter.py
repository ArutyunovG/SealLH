from torch.utils.data import Dataset

from seallh.helpers.dataset.adapters.category_mapping_adapter import CategoryMappingAdapter


class ToSingleClassAdapter(CategoryMappingAdapter):

    def __init__(self, dataset: Dataset, class_name: str = "object"):
        categories = dataset.categories
        mapping = {i: 0 for i in range(len(categories))}
        names = {0: class_name}
        super().__init__(dataset, mapping=mapping, names=names)
