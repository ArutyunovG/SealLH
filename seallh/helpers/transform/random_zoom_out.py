from typing import Any

import cv2
import numpy as np

from albumentations.core.transforms_interface import DualTransform
from albumentations.core.type_definitions import Targets

from seallh.helpers.transform.albumentations_transform import AlbumentationsTransform


class _RandomZoomOut(DualTransform):
    """Place the image on a larger constant-colored canvas at a random position.

    Simulates zooming out: the original image is pasted onto a bigger canvas
    filled with a constant value, shifting every bbox by the paste offset.
    Useful for teaching detectors to recognize objects at smaller scales.

    Args:
        max_zoom_ratio: Upper bound of the canvas-size multiplier relative to
            the input. The sampled ratio is drawn uniformly from
            [1, max_zoom_ratio]; the canvas area is up to max_zoom_ratio**2
            times the original area. Default: 4.0.
        fill: Scalar constant used to pad the canvas (applied to every
            channel). Default: 0.0.
        resize_back: If True, resize the padded canvas back to the original
            (height, width) so the transform preserves the input resolution.
            Bboxes are scaled accordingly. Default: True.
        interpolation: OpenCV interpolation flag used when ``resize_back`` is
            True (e.g. ``cv2.INTER_LINEAR``, ``cv2.INTER_NEAREST``,
            ``cv2.INTER_CUBIC``, ``cv2.INTER_AREA``). Default:
            ``cv2.INTER_LINEAR``.
        p: Probability of applying the transform. Default: 0.5.

    Targets:
        image, bboxes

    Image types:
        uint8, float32
    """

    _targets = (Targets.IMAGE, Targets.BBOXES)

    def __init__(
        self,
        max_zoom_ratio: float = 4.0,
        fill: float = 0.0,
        resize_back: bool = True,
        interpolation: int = cv2.INTER_LINEAR,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        if max_zoom_ratio < 1.0:
            raise ValueError(f"max_zoom_ratio must be >= 1.0, got {max_zoom_ratio}")
        self.max_zoom_ratio = float(max_zoom_ratio)
        self.fill = float(fill)
        self.resize_back = bool(resize_back)
        self.interpolation = int(interpolation)


    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        height, width = params["shape"][:2]

        eps = 1e-5
        if self.max_zoom_ratio - 1.0 < eps:
            return {"ratio": 1.0, "top": 0, "left": 0}

        ratio = self.py_random.uniform(1.0, self.max_zoom_ratio)
        new_h = int(height * ratio)
        new_w = int(width * ratio)

        top = int(self.py_random.uniform(0.0, float(new_h - height)))
        left = int(self.py_random.uniform(0.0, float(new_w - width)))
        return {"ratio": ratio, "top": top, "left": left}


    def _make_canvas(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype,
        ratio: float,
    ) -> np.ndarray:
        height, width = shape[:2]
        new_h, new_w = int(height * ratio), int(width * ratio)
        new_shape = (new_h, new_w) if len(shape) == 2 else (new_h, new_w, shape[2])
        return np.full(new_shape, self.fill, dtype=dtype)

    def apply(
        self,
        img: np.ndarray,
        ratio: float,
        top: int,
        left: int,
        **params: Any,
    ) -> np.ndarray:
        if ratio == 1.0:
            return img
        h, w = img.shape[:2]
        canvas = self._make_canvas(img.shape, img.dtype, ratio)
        canvas[top : top + h, left : left + w] = img
        if self.resize_back:
            canvas = cv2.resize(canvas, (w, h), interpolation=self.interpolation)
        return canvas


    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        ratio: float,
        top: int,
        left: int,
        **params: Any,
    ) -> np.ndarray:
        if ratio == 1.0 or len(bboxes) == 0:
            return bboxes
        # Internal bbox format is normalized [x_min, y_min, x_max, y_max, ...].
        # We rescale the normalized coords to the new canvas.
        h, w = params["shape"][:2]
        new_h, new_w = int(h * ratio), int(w * ratio)
        dx = left / new_w
        dy = top / new_h
        sx = w / new_w
        sy = h / new_h

        out = bboxes.copy().astype(np.float64, copy=False)
        out[:, 0] = bboxes[:, 0] * sx + dx
        out[:, 1] = bboxes[:, 1] * sy + dy
        out[:, 2] = bboxes[:, 2] * sx + dx
        out[:, 3] = bboxes[:, 3] * sy + dy
        return out.astype(bboxes.dtype, copy=False)


RandomZoomOut = lambda **cfg: AlbumentationsTransform(_RandomZoomOut, **cfg)
