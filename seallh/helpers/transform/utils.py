import numpy as np


def pairwise_iof(a, b):
    """Fraction of each box in `a` that lies inside each box in `b`.

    IoF(a_i, b_j) = area(a_i ∩ b_j) / area(a_i). Shape: (len(a), len(b)).
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1e-10)


def pairwise_jaccard(a, b):
    """IoU between each box in `a` and each box in `b`. Shape: (len(a), len(b))."""
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    union = area_a[:, np.newaxis] + area_b[np.newaxis, :] - area_i
    return area_i / np.maximum(union, 1e-10)


def pairwise_sample_coverage(a, b):
    """Fraction of each box in `b` (the 'sample' / crop) covered by each box in `a`.

    sample_coverage(a_i, b_j) = area(a_i ∩ b_j) / area(b_j). Shape: (len(a), len(b)).
    Note: SSD defines sample_coverage as area(intersection) / area(sample),
    where 'sample' = the crop. So we normalize by b's area.
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / np.maximum(area_b[np.newaxis, :], 1e-10)
