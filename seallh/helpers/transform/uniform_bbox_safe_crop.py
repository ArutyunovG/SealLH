import math
import random
import numpy as np

from typing import Dict, Tuple, Any, Optional
from albumentations.augmentations.crops.transforms import BaseCrop
from albumentations.core.bbox_utils import denormalize_bboxes

from seallh.helpers.transform.albumentations_transform import AlbumentationsTransform
from seallh.helpers.transform.utils import (pairwise_iof,
                                            pairwise_jaccard,
                                            pairwise_sample_coverage)


class _UniformBBoxSafeCrop(BaseCrop):
    """
    Random crop with continuous uniform sampling of scale and aspect ratio,
    accepted only if at least `min_boxes_kept` ground-truth boxes satisfy a
    configurable overlap constraint with the crop. Generalizes the per-sampler
    behavior of Wei Liu's Caffe-SSD `batch_sampler` and the SCRFD bbox-safe
    crop into a single transform.

    Crop geometry (pixel coords):
        base_side  L = scale * min(H, W)
        width      w = round(L * sqrt(aspect_ratio))
        height     h = round(L / sqrt(aspect_ratio))

    The acceptance test counts GT boxes whose `metric` value against the crop
    falls in [min_threshold, max_threshold], and accepts the crop iff the count
    is at least `min_boxes_kept`.

    Supported metrics:
        'iof'              — area(box ∩ crop) / area(box).
                             SSD's `object_coverage`. Use threshold=1.0 for
                             strict containment (SCRFD default).
        'jaccard'          — area(box ∩ crop) / area(box ∪ crop).
                             SSD's `jaccard_overlap`. Standard IoU.
        'sample_coverage'  — area(box ∩ crop) / area(crop).
                             SSD's `sample_coverage`. Fraction of the crop
                             covered by a given box.

    Args:
        scale: (min, max) fraction of min(H, W) for the crop's base side.
            Sampled uniformly in linear space. SSD default: (0.3, 1.0).
        ratio: (min, max) aspect ratio w / h, sampled uniformly in linear space.
            SSD default: (0.5, 2.0).
        metric: 'iof', 'jaccard', or 'sample_coverage'.
        min_threshold: lower bound on the metric for a box to count (inclusive).
            None means no lower bound. Default 1.0 (strict containment, SCRFD).
        max_threshold: upper bound on the metric for a box to count (inclusive).
            None means no upper bound.
        min_boxes_kept: minimum number of boxes satisfying the threshold window.
        max_trials: rejection-sampling budget before falling back.
        p: probability of applying the transform.

    SSD batch_sampler correspondence (for OneOf-style composition):
        use_original_image       → metric='iof', min_threshold=None, max_threshold=None, p=0 (or just skip)
        min_jaccard_overlap=0.1  → metric='jaccard', min_threshold=0.1
        min_jaccard_overlap=0.3  → metric='jaccard', min_threshold=0.3
        min_jaccard_overlap=0.5  → metric='jaccard', min_threshold=0.5
        min_jaccard_overlap=0.7  → metric='jaccard', min_threshold=0.7
        min_jaccard_overlap=0.9  → metric='jaccard', min_threshold=0.9
        max_jaccard_overlap=1.0  → metric='jaccard', max_threshold=1.0

    SCRFD-style configuration:
        metric='iof', min_threshold=1.0, min_boxes_kept=N
    """

    _METRIC_FNS = {
        "iof": pairwise_iof,
        "jaccard": pairwise_jaccard,
        "sample_coverage": pairwise_sample_coverage,
    }

    def __init__(
        self,
        scale: Tuple[float, float] = (0.3, 1.0),
        ratio: Tuple[float, float] = (0.5, 2.0),
        metric: str = "iof",
        min_threshold: Optional[float] = 1.0,
        max_threshold: Optional[float] = None,
        min_boxes_kept: int = 1,
        max_trials: int = 50,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        assert 0 < scale[0] <= scale[1]
        assert 0 < ratio[0] <= ratio[1]
        assert metric in self._METRIC_FNS, f"metric must be one of {list(self._METRIC_FNS)}"
        if min_threshold is not None:
            assert 0.0 <= min_threshold <= 1.0
        if max_threshold is not None:
            assert 0.0 <= max_threshold <= 1.0
        if min_threshold is not None and max_threshold is not None:
            assert min_threshold <= max_threshold
        assert min_boxes_kept >= 0
        assert max_trials >= 1

        self.scale = tuple(scale)
        self.ratio = tuple(ratio)
        self.metric = metric
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.min_boxes_kept = min_boxes_kept
        self.max_trials = max_trials

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def _count_satisfying(self, boxes_px: np.ndarray, roi: np.ndarray) -> int:
        """Count how many GT boxes satisfy the threshold window for the given crop."""
        metric_fn = self._METRIC_FNS[self.metric]
        values = metric_fn(boxes_px, roi)  # shape (N, 1)

        mask = np.ones(values.shape, dtype=bool)
        if self.min_threshold is not None:
            mask &= values >= self.min_threshold
        if self.max_threshold is not None:
            mask &= values <= self.max_threshold
        return int(mask.sum())

    def get_params_dependent_on_data(
        self,
        params: Dict[str, Any],
        data: Dict[str, Any],
    ) -> Dict[str, Any]:

        img_h, img_w = params["shape"][:2]
        bboxes = data.get("bboxes", [])

        # Denormalize GT boxes to pixel coords for metric computation.
        # AlbumentationsTransform normalizes bboxes to [0, 1] before calling us.
        if len(bboxes) > 0:
            bboxes_arr = np.array(bboxes, dtype=np.float32)
            if bboxes_arr.ndim == 1:
                bboxes_arr = bboxes_arr.reshape(1, -1)
            boxes_px = denormalize_bboxes(bboxes_arr, (img_h, img_w))[:, :4]
        else:
            boxes_px = None

        no_constraint = self.min_threshold is None and self.max_threshold is None

        if no_constraint or boxes_px is None:
            return {"crop_coords": (0, 0, img_w, img_h)}

        for _ in range(self.max_trials):
            s = random.uniform(*self.scale)
            ar = random.uniform(*self.ratio)

            base = s * min(img_h, img_w)
            w = int(round(base * math.sqrt(ar)))
            h = int(round(base / math.sqrt(ar)))

            if not (0 < w <= img_w and 0 < h <= img_h):
                continue

            j = random.randint(0, img_w - w)
            i = random.randint(0, img_h - h)

            roi = np.array([[j, i, j + w, i + h]], dtype=np.float32)
            if self._count_satisfying(boxes_px, roi) >= self.min_boxes_kept:
                return {"crop_coords": (j, i, j + w, i + h)}

        # Fallback: unmodified image.
        return {"crop_coords": (0, 0, img_w, img_h)}


UniformBBoxSafeCrop = lambda **cfg: AlbumentationsTransform(_UniformBBoxSafeCrop, **cfg)
