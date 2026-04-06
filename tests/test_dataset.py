"""Tests for TDE datasets."""

from __future__ import annotations

import torch

from anima_tde.dataset import EvDETDataset, VOCDataset, collate_fn


class TestVOCDataset:
    """Test VOC dataset loader."""

    def test_init_missing_dir(self):
        """Dataset should initialize even with missing directory (0 samples)."""
        ds = VOCDataset(
            root="/nonexistent/path",
            split="train",
            img_size=(320, 320),
            timesteps=4,
        )
        assert len(ds) == 0

    def test_collate_fn_empty(self):
        """Collate function handles empty batch."""
        batch = [
            {
                "image": torch.randn(3, 320, 320),
                "targets": torch.zeros((0, 6)),
                "image_path": "test.jpg",
            },
            {
                "image": torch.randn(3, 320, 320),
                "targets": torch.zeros((0, 6)),
                "image_path": "test2.jpg",
            },
        ]
        result = collate_fn(batch)
        assert result["images"].shape == (2, 3, 320, 320)
        assert result["targets"].shape[1] == 6

    def test_collate_fn_with_targets(self):
        """Collate function handles targets with batch index."""
        batch = [
            {
                "image": torch.randn(3, 320, 320),
                "targets": torch.tensor([[0, 5, 0.5, 0.5, 0.1, 0.1]]),
                "image_path": "test.jpg",
            },
            {
                "image": torch.randn(3, 320, 320),
                "targets": torch.tensor([
                    [0, 3, 0.3, 0.3, 0.2, 0.2],
                    [0, 7, 0.7, 0.7, 0.1, 0.1],
                ]),
                "image_path": "test2.jpg",
            },
        ]
        result = collate_fn(batch)
        assert result["images"].shape == (2, 3, 320, 320)
        assert result["targets"].shape == (3, 6)
        # Batch indices should be set
        assert result["targets"][0, 0] == 0  # first image
        assert result["targets"][1, 0] == 1  # second image
        assert result["targets"][2, 0] == 1  # second image


class TestEvDETDataset:
    """Test EvDET200K dataset loader."""

    def test_init_missing_dir(self):
        ds = EvDETDataset(
            root="/nonexistent/path",
            split="train",
            img_size=(304, 240),
            timesteps=4,
        )
        assert len(ds) == 0
