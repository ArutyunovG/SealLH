from pathlib import Path
import json
from PIL import Image

import numpy as np
from torch.utils.data import Dataset


class COCODataset(Dataset):

    """
    PyTorch Dataset for COCO-style annotations.

    Args:
        images_dir (str | Path): Path to directory with images
        annotation_file (str | Path): Path to COCO JSON annotations
    """

    def __init__(self, root_dir, split, images_dir, annotation_file, single_class=False):

        self.root_dir = Path(root_dir).resolve()
        self.single_class = single_class

        self.images_dir = self.root_dir / images_dir
        assert self.images_dir.is_dir(), f"Images directory not found: {self.images_dir}"
        self.annotation_file = self.root_dir / annotation_file
        assert self.annotation_file.is_file(), f"Annotation file not found: {self.annotation_file}"

        self.split = split

        with open(self.annotation_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = {int(img["id"]): img for img in coco["images"]}

        sorted_cats = sorted(coco["categories"], key=lambda x: int(x['id']))

        if self.single_class:
            self.cat_mapping = {int(cat["id"]): 0 for cat in sorted_cats}
            self.categories = ["object"]
        else:
            self.cat_mapping = {int(cat["id"]): idx for idx, cat in enumerate(sorted_cats)}
            self.categories = [cat['name'] for cat in coco["categories"]]

        self.img_to_anns = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            self.img_to_anns.setdefault(img_id, []).append(ann)

        self.image_ids = list(self.images.keys())


    def __len__(self):
        return len(self.image_ids)


    def __getitem__(self, idx):

        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        img_path = self.images_dir / img_info["file_name"]
        # Ensure image has 3 channels (RGB). Some COCO images may be grayscale.
        image = Image.open(str(img_path)).convert('RGB')
        image = np.array(image)

        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        annotations = self.img_to_anns.get(img_id, [])

        h, w = image.shape[:2]

        labels = []
        for ann in annotations:
            if "bbox" not in ann:
                continue
            labels.append(self.cat_mapping.get(int(ann["category_id"])))

        data_dct = {
            "image": image,
            "img_id": img_id,
            "img_shape": (h, w),
            "bboxes": [ann["bbox"] for ann in annotations if "bbox" in ann],
            "labels": labels,
        }

        return data_dct
