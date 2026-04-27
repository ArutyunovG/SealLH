import math
from typing import Any

import numpy as np
from albumentations import DualTransform
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.core.bbox_utils import denormalize_bboxes, normalize_bboxes
from albumentations.core.transforms_interface import Targets

from seallh.helpers.transform.albumentations_transform import AlbumentationsTransform


class _PadToAspectRatio(DualTransform):
    """Pad the minimal number of pixels so the image reaches a target W/H aspect ratio."""

    _targets = (Targets.IMAGE, Targets.BBOXES)

    def __init__(
        self,
        aspect_ratio: float = 1.0,
        position: str = "center",
        border_mode: int = 0,
        fill: float = 0,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.aspect_ratio = float(aspect_ratio)
        assert self.aspect_ratio > 0
        self.position = position
        self.border_mode = border_mode
        self.fill = fill

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        h, w = params["shape"][:2]
        current = w / h

        if current < self.aspect_ratio:
            new_w = int(math.ceil(self.aspect_ratio * h))
            new_h = h
        elif current > self.aspect_ratio:
            new_h = int(math.ceil(w / self.aspect_ratio))
            new_w = w
        else:
            new_h, new_w = h, w

        pad_h = new_h - h
        pad_w = new_w - w

        h_top, h_bottom, w_left, w_right = fgeometric.adjust_padding_by_position(
            h_top=pad_h // 2,
            h_bottom=pad_h - pad_h // 2,
            w_left=pad_w // 2,
            w_right=pad_w - pad_w // 2,
            position=self.position,
            py_random=self.py_random,
        )

        return {
            "pad_top": h_top,
            "pad_bottom": h_bottom,
            "pad_left": w_left,
            "pad_right": w_right,
        }

    def apply(self, img: np.ndarray, pad_top: int = 0, pad_bottom: int = 0,
              pad_left: int = 0, pad_right: int = 0, **params: Any) -> np.ndarray:
        return fgeometric.pad_with_params(
            img, pad_top, pad_bottom, pad_left, pad_right,
            border_mode=self.border_mode, value=self.fill,
        )

    def apply_to_bboxes(self, bboxes: np.ndarray, pad_top: int = 0, pad_bottom: int = 0,
                        pad_left: int = 0, pad_right: int = 0, **params: Any) -> np.ndarray:
        image_shape = params["shape"][:2]
        bboxes_denorm = denormalize_bboxes(bboxes, params["shape"])
        result = fgeometric.pad_bboxes(
            bboxes_denorm, pad_top, pad_bottom, pad_left, pad_right,
            self.border_mode, image_shape=image_shape,
        )
        rows, cols = image_shape
        return normalize_bboxes(result, (rows + pad_top + pad_bottom, cols + pad_left + pad_right))

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "aspect_ratio", "position", "border_mode", "fill"


PadToAspectRatio = lambda **cfg: AlbumentationsTransform(_PadToAspectRatio, **cfg)
