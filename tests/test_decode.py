"""Tests for detection decoder."""

from __future__ import annotations

import torch

from anima_tde.decode import decode_predictions


class TestDecode:
    """Test prediction decoder."""

    def test_decode_shape(self):
        """Decoder should return one result per batch element."""
        preds = [
            torch.randn(2, 75, 16, 16),  # scale 1
            torch.randn(2, 75, 8, 8),    # scale 2
            torch.randn(2, 75, 4, 4),    # scale 3
        ]
        results = decode_predictions(preds, num_classes=20, conf_threshold=0.5)
        assert len(results) == 2  # batch size 2
        for r in results:
            assert "boxes" in r
            assert "scores" in r
            assert "class_ids" in r
            assert r["boxes"].shape[1] == 4

    def test_decode_high_threshold(self):
        """Very high threshold should produce no detections."""
        preds = [torch.randn(1, 75, 8, 8)]
        results = decode_predictions(preds, conf_threshold=0.9999)
        # Most random predictions won't pass high threshold
        assert len(results) == 1
        # Not guaranteed to be 0 but should be very few
        assert results[0]["boxes"].shape[0] < 50

    def test_decode_empty_predictions(self):
        """Empty predictions should return empty results."""
        preds = [torch.zeros(1, 75, 4, 4)]  # zeros -> sigmoid(0)=0.5
        results = decode_predictions(preds, conf_threshold=0.99)
        assert len(results) == 1
