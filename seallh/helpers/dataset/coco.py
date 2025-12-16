from pathlib import Path
import json
import cv2
import torch
from torch.utils.data import Dataset


class COCODataset(Dataset):

    """
    PyTorch Dataset for COCO-style annotations.

    Args:
        images_dir (str | Path): Path to directory with images
        annotation_file (str | Path): Path to COCO JSON annotations
    """

    def __init__(self, root_dir, split, images_dir, annotation_file):

        self.root_dir = Path(root_dir).resolve()

        self.images_dir = self.root_dir / images_dir
        assert self.images_dir.is_dir(), f"Images directory not found: {self.images_dir}"
        self.annotation_file = self.root_dir / annotation_file
        assert self.annotation_file.is_file(), f"Annotation file not found: {self.annotation_file}"

        self.split = split

        with open(self.annotation_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = {int(img["id"]): img for img in coco["images"]}

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
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        annotations = self.img_to_anns.get(img_id, [])

        target = {
            "image_id": torch.tensor(img_id),
            "annotations": annotations,
        }

        return image, target
