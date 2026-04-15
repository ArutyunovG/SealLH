import logging

from seallh.helpers.dataset.modifiers.base_modifier import DatasetModifier
from seallh.helpers.dataset.coco import COCODataset

logger = logging.getLogger("seallh.helpers.dataset.modifiers.coco_empty_filter")


class COCOEmptyFilter(DatasetModifier):
    """Remove images that have no annotations.

    Args:
        None
    """

    def __call__(self, dataset: COCODataset) -> COCODataset:

        assert isinstance(dataset, COCODataset), f"COCOEmptyFilter expects COCODataset, got {type(dataset)}"

        kept_image_ids = [
            img_id for img_id in dataset.image_ids
            if dataset.img_to_anns.get(img_id)
        ]

        original_len = len(dataset.image_ids)
        dataset.image_ids = kept_image_ids

        logger.info(
            "COCOEmptyFilter: %d -> %d images",
            original_len, len(kept_image_ids),
        )

        return dataset
