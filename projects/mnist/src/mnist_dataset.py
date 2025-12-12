import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MNISTDataset(Dataset):

    """
    PyTorch Dataset for MNIST-style folders using OpenCV:

    root/
      train/
        0/, 1/, ..., 9/   # PNGs
      test/
        0/, 1/, ..., 9/

    Args:
        root_dir: Root path containing 'train' and/or 'test' subfolders.
        split: "train" or "test".
    """

    def __init__(self,
                 root_dir: str,
                 split: str):

        assert split in ("train", "test"), "split must be 'train' or 'test'"
        self.root_dir = root_dir
        self.split = split
        self.samples: List[Tuple[str, int]] = self._collect_samples()


    def _collect_samples(self) -> List[Tuple[str, int]]:

        split_dir = os.path.join(self.root_dir, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split folder not found: {split_dir}")

        samples: List[Tuple[str, int]] = []
        for label in sorted(os.listdir(split_dir)):
            label_dir = os.path.join(split_dir, label)
            if not os.path.isdir(label_dir):
                continue
            try:
                y = int(label)
            except ValueError:
                # skip non-numeric class folders (e.g., 'unknown')
                continue
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(".png"):
                    samples.append((os.path.join(label_dir, fname), y))
        if not samples:
            raise RuntimeError(f"No PNG files found under {split_dir}")
        return samples


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, index: int):
        path, target = self.samples[index]
        # Read grayscale with OpenCV: shape (H, W), dtype=uint8
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        # Flatten to 784 vector
        img = img.flatten()
        assert img.shape == (28 * 28,), f"Unexpected image shape: {img.shape}"
        # Convert to float32 tensor in [0,1]
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img)
        return img, target
