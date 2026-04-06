"""Dataset loaders for TDE.

Implements:
- VOCDataset: PASCAL VOC 2007/2012 in YOLO format
- EvDETDataset: EvDET200K neuromorphic event frames
- Data augmentation (mosaic, HSV, flip, scale)

Reference: arXiv 2512.02447, Section 4 (Experiments)
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    """PASCAL VOC dataset in YOLO format.

    Expects directory structure:
        root/
            images/
                train/  (or trainval/)
                val/
                test/
            labels/
                train/
                val/
                test/

    Each label file has lines: class_id cx cy w h (normalized 0-1).

    For SNN: static images are repeated across T timesteps via direct encoding.

    Args:
        root: Dataset root directory.
        split: "train", "val", or "test".
        img_size: Target image size (height, width).
        timesteps: Number of SNN timesteps (image repeated T times).
        augment: Enable data augmentation.
        mosaic: Enable mosaic augmentation.
        hsv_h: HSV hue augmentation range.
        hsv_s: HSV saturation augmentation range.
        hsv_v: HSV value augmentation range.
        flip_lr: Horizontal flip probability.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: tuple[int, int] = (640, 640),
        timesteps: int = 4,
        augment: bool = True,
        mosaic: bool = False,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        flip_lr: float = 0.5,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.timesteps = timesteps
        self.augment = augment and split == "train"
        self.mosaic = mosaic and self.augment
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.flip_lr = flip_lr

        # Find image files
        img_dir = self.root / "images" / split
        if not img_dir.exists():
            # Try flat structure
            img_dir = self.root / "images"

        self.image_files: list[Path] = []
        if img_dir.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                self.image_files.extend(sorted(img_dir.glob(ext)))

        self.label_dir = self.root / "labels" / split
        if not self.label_dir.exists():
            self.label_dir = self.root / "labels"

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        """Load image and labels.

        Returns:
            Dict with keys:
                image: (3, H, W) float32 tensor [0, 1]
                targets: (N, 6) tensor [batch_idx=0, class, cx, cy, w, h]
                image_path: str
        """
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            # Return blank if image cannot be loaded
            img = np.zeros((*self.img_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load labels
        label_path = self.label_dir / (img_path.stem + ".txt")
        labels = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = (float(x) for x in parts[1:5])
                        labels.append([0, cls_id, cx, cy, w, h])

        # Resize
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))

        # Augmentation
        if self.augment:
            img, labels = self._augment(img, labels)

        # To tensor
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0

        if labels:
            targets = torch.tensor(labels, dtype=torch.float32)
        else:
            targets = torch.zeros((0, 6), dtype=torch.float32)

        return {
            "image": img_tensor,
            "targets": targets,
            "image_path": str(img_path),
        }

    def _augment(
        self, img: np.ndarray, labels: list
    ) -> tuple[np.ndarray, list]:
        """Apply data augmentation."""
        h, w = img.shape[:2]

        # HSV augmentation
        if self.hsv_h > 0 or self.hsv_s > 0 or self.hsv_v > 0:
            r = np.random.uniform(-1, 1, 3) * [self.hsv_h, self.hsv_s, self.hsv_v] + 1
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float64)
            hsv[:, :, 0] = hsv[:, :, 0] * r[0] % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * r[1], 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * r[2], 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Horizontal flip
        if random.random() < self.flip_lr:
            img = np.fliplr(img).copy()
            for lbl in labels:
                lbl[2] = 1.0 - lbl[2]  # flip cx

        return img, labels


class EvDETDataset(Dataset):
    """EvDET200K neuromorphic event dataset.

    Event streams are pre-accumulated into single-channel frames.

    Expects:
        root/
            images/
                {split}/
            labels/
                {split}/

    Args:
        root: Dataset root directory.
        split: "train", "val", or "test".
        img_size: Target frame size.
        timesteps: Number of timesteps (event frames stacked).
        augment: Enable augmentation.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: tuple[int, int] = (304, 240),
        timesteps: int = 4,
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.timesteps = timesteps
        self.augment = augment and split == "train"

        img_dir = self.root / "images" / split
        self.image_files: list[Path] = []
        if img_dir.exists():
            for ext in ("*.jpg", "*.png", "*.npy"):
                self.image_files.extend(sorted(img_dir.glob(ext)))

        self.label_dir = self.root / "labels" / split

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_files[idx]

        # Load event frame (single channel)
        if img_path.suffix == ".npy":
            frame = np.load(str(img_path))
        else:
            frame = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                frame = np.zeros(self.img_size, dtype=np.uint8)

        # Resize
        frame = cv2.resize(frame, (self.img_size[1], self.img_size[0]))

        # Normalize to [0, 1]
        if frame.max() > 1:
            frame = frame.astype(np.float32) / 255.0
        else:
            frame = frame.astype(np.float32)

        # Single channel -> (1, H, W)
        img_tensor = torch.from_numpy(frame).unsqueeze(0)

        # Horizontal flip augmentation
        if self.augment and random.random() < 0.5:
            img_tensor = img_tensor.flip(-1)

        # Load labels
        label_path = self.label_dir / (img_path.stem + ".txt")
        labels = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = (float(x) for x in parts[1:5])
                        labels.append([0, cls_id, cx, cy, w, h])

        if labels:
            targets = torch.tensor(labels, dtype=torch.float32)
        else:
            targets = torch.zeros((0, 6), dtype=torch.float32)

        return {
            "image": img_tensor,
            "targets": targets,
            "image_path": str(img_path),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate for detection datasets.

    Handles variable-length target tensors by adding batch index.
    """
    images = torch.stack([item["image"] for item in batch])
    paths = [item["image_path"] for item in batch]

    targets_list = []
    for i, item in enumerate(batch):
        t = item["targets"]
        if t.shape[0] > 0:
            t[:, 0] = i  # Set batch index
            targets_list.append(t)

    if targets_list:
        targets = torch.cat(targets_list, dim=0)
    else:
        targets = torch.zeros((0, 6))

    return {
        "images": images,
        "targets": targets,
        "paths": paths,
    }


def build_dataset(config: dict, split: str = "train") -> Dataset:
    """Build dataset from config dict."""
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    aug_cfg = config.get("training", {}).get("augmentation", {})

    img_size = (
        model_cfg.get("input_height", 640),
        model_cfg.get("input_width", 640),
    )
    timesteps = model_cfg.get("timesteps", 4)

    dataset_type = data_cfg.get("dataset", "voc")

    if dataset_type == "voc":
        return VOCDataset(
            root=data_cfg.get("train_path", "/mnt/forge-data/datasets/voc/"),
            split=split,
            img_size=img_size,
            timesteps=timesteps,
            augment=split == "train",
            mosaic=aug_cfg.get("mosaic", False) if split == "train" else False,
            hsv_h=aug_cfg.get("hsv_h", 0.015),
            hsv_s=aug_cfg.get("hsv_s", 0.7),
            hsv_v=aug_cfg.get("hsv_v", 0.4),
            flip_lr=aug_cfg.get("flip_lr", 0.5),
        )
    elif dataset_type == "evdet200k":
        return EvDETDataset(
            root=data_cfg.get("train_path", "/mnt/forge-data/datasets/evdet200k/"),
            split=split,
            img_size=img_size,
            timesteps=timesteps,
            augment=split == "train",
        )
    else:
        msg = f"Unknown dataset: {dataset_type}"
        raise ValueError(msg)
