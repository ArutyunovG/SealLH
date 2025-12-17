import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from seallh.helpers.dataset.coco import COCODataset as SeallhCOCO

from nanodet.data.dataset.base import BaseDataset


class SeallhCOCOToNanoDetDataset(BaseDataset):
    """Adapter that wraps a seallh COCODataset and exposes NanoDet BaseDataset API.

    Args:
        images_dir (str|Path): passed to seallh COCODataset
        annotation_file (str|Path): passed to seallh COCODataset
        input_size, pipeline, keep_ratio, use_instance_mask, use_seg_mask,
        use_keypoint, load_mosaic, mode, multi_scale: same as NanoDet BaseDataset params
        class_names (list[str], optional): list of class names to map category ids to labels.
        seallh_coco_cls (callable, optional): class to instantiate seallh dataset (defaults
            to importing at runtime to avoid hard dependency during import time).
    """

    def __init__(
        self,
        root_dir,
        split,
        images_dir,
        annotation_file,
        input_size: Tuple[int, int] = (320, 320),
        pipeline=None,
        keep_ratio: bool = True,
        use_instance_mask: bool = False,
        use_seg_mask: bool = False,
        use_keypoint: bool = False,
        load_mosaic: bool = False,
        multi_scale: Optional[Tuple[float, float]] = None,
        class_names: Optional[List[str]] = None,
        **kwargs,
    ):
        # Map incoming `split` (train/validation/test) to NanoDet BaseDataset `mode`
        if split in ("train", "training"):
            mode = "train"
        elif split in ("validation", "val"):
            mode = "val"
        elif split == "test":
            mode = "test"
        else:
            mode = "train"

        # BaseDataset initializer expects img_path and ann_path as strings
        super().__init__(
            str(Path(root_dir).resolve() / images_dir),
            str(Path(root_dir).resolve() / annotation_file),
            input_size,
            pipeline or {},
            keep_ratio=keep_ratio,
            use_instance_mask=use_instance_mask,
            use_seg_mask=use_seg_mask,
            use_keypoint=use_keypoint,
            load_mosaic=load_mosaic,
            mode=mode,
            multi_scale=multi_scale,
        )

        # Instantiate seallh COCO dataset (it returns image, target) on index
        self.seallh_coco = SeallhCOCO(root_dir=root_dir,
                                      split=split,
                                      images_dir=images_dir,
                                      annotation_file=annotation_file)

        # Build id -> position map for quick lookup
        # seallh dataset exposes `image_ids` or images mapping; try both
        self.id_to_pos = {int(i): p for p, i in enumerate(self.seallh_coco.image_ids)}

        # Build category id -> label mapping. Try to read categories from annotation file.
        try:
            ann_path = Path(Path(root_dir).resolve() / annotation_file)
            with open(ann_path, "r", encoding="utf-8") as f:
                coco = json.load(f)
            categories = coco.get("categories", [])
            cid2name = {c["id"]: c["name"] for c in categories}
        except Exception:
            cid2name = {}

        if class_names:
            # Map COCO category ids to indices in provided class_names list
            self.cat2label = {
                cid: class_names.index(name)
                for cid, name in cid2name.items()
                if name in class_names
            }
        else:
            # Default mapping: enumerate sorted category ids
            sorted_cids = sorted(cid2name.keys())
            self.cat2label = {cid: i for i, cid in enumerate(sorted_cids)}


    def get_data_info(self, ann_path):
        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)
        img_info = coco.get("images", [])
        img_info = sorted(img_info, key=lambda x: int(x["id"]))
        return img_info


    def get_train_data(self, idx):
        info = self.get_per_img_info(idx)
        img_id = int(info["id"])
        pos = self.id_to_pos.get(img_id)
        if pos is None:
            raise IndexError(f"Image id {img_id} not found in Seallh dataset")

        img, target = self.seallh_coco[pos]

        anns = target.get("annotations", [])
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        for ann in anns:
            # ann expected to contain bbox in [x,y,w,h]
            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            if ann.get("area", 0) <= 0 or w < 1 or h < 1:
                continue
            cid = ann.get("category_id")
            if cid not in self.cat2label:
                # skip unknown categories
                continue
            bbox = [x, y, x + w, y + h]
            if ann.get("iscrowd", False) or ann.get("ignore", False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[cid])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        meta = dict(
            img=img,
            img_info=info,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
        )

        if self.use_instance_mask:
            # seallh target may include masks, but adapter doesn't implement masks translation yet
            meta["gt_masks"] = []
        if self.use_keypoint:
            meta["gt_keypoints"] = np.zeros((0, 51), dtype=np.float32)

        input_size = self.input_size
        if self.multi_scale:
            input_size = self.get_random_size(self.multi_scale, input_size)

        meta = self.pipeline(self, meta, input_size)

        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1))
        return meta


    def get_val_data(self, idx):
        # For now, same behavior as train
        return self.get_train_data(idx)


    def get_per_img_info(self, idx):
        img_info = self.data_info[idx]
        file_name = img_info["file_name"]
        height = img_info["height"]
        width = img_info["width"]
        id = img_info["id"]
        if not isinstance(id, int):
            raise TypeError("Image id must be int.")
        info = {"file_name": file_name, "height": height, "width": width, "id": id}
        return info
