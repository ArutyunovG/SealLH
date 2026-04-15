import logging
from typing import List

from torch.utils.data import Dataset

from seallh.helpers.dataset.modifiers.base_modifier import DatasetModifier
from seallh.helpers.dataset.coco import COCODataset

logger = logging.getLogger("seallh.helpers.dataset.modifiers.coco_class_filter")


class COCOClassFilter(DatasetModifier):
    """Keep only images that contain at least one instance of the given classes.

    Works with COCODataset by inspecting its annotation metadata — no image
    loading required.  Annotations for classes not in the list are removed.

    Args:
        class_names: List of category names to keep (e.g. ["person", "car"]).
    """

    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    def __call__(self, dataset: COCODataset) -> COCODataset:

        assert isinstance(dataset, COCODataset), f"COCOClassFilter expects COCODataset, got {type(dataset)}"

        target_names = set(self.class_names)

        cat_ids = set()
        for raw_cat_id, mapped_idx in dataset.cat_mapping.items():
            if dataset.categories[mapped_idx] in target_names:
                cat_ids.add(raw_cat_id)

        found_names = {dataset.categories[dataset.cat_mapping[cid]] for cid in cat_ids}
        missing = target_names - found_names
        if missing:
            raise ValueError(
                f"Classes {missing} not found in dataset categories: "
                f"{dataset.categories}"
            )

        kept_image_ids = []
        filtered_img_to_anns = {}
        for img_id in dataset.image_ids:
            anns = dataset.img_to_anns.get(img_id, [])
            matching = [a for a in anns if a["category_id"] in cat_ids]
            if matching:
                kept_image_ids.append(img_id)
                filtered_img_to_anns[img_id] = matching

        original_len = len(dataset.image_ids)
        dataset.image_ids = kept_image_ids
        dataset.img_to_anns = filtered_img_to_anns

        # Remap categories to only the kept classes with contiguous indices
        new_categories = [name for name in dataset.categories if name in target_names]
        new_cat_mapping = {}
        for raw_cat_id, mapped_idx in dataset.cat_mapping.items():
            if raw_cat_id in cat_ids:
                new_cat_mapping[raw_cat_id] = new_categories.index(dataset.categories[mapped_idx])
        dataset.categories = new_categories
        dataset.cat_mapping = new_cat_mapping

        logger.info(
            "COCOClassFilter(%s): %d -> %d images",
            self.class_names, original_len, len(kept_image_ids),
        )

        return dataset
