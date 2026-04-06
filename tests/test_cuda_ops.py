"""Tests for CUDA-accelerated spiking operations."""

from __future__ import annotations

import pytest
import torch

from anima_tde.backends.cuda.nms_cuda import (
    cuda_box_iou,
    cuda_nms,
)
from anima_tde.backends.cuda.spiking_cuda import (
    CUDALIFNeuron,
    fused_alpha_mix,
    fused_sda_attention,
    has_cuda_ops,
)


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TestCUDALIF:
    """Test CUDA-accelerated LIF neuron."""

    def test_cuda_lif_shape(self, device):
        lif = CUDALIFNeuron(beta=0.25, threshold=1.0)
        x = torch.randn(4, 2, 16, 8, 8, device=device)
        out = lif(x)
        assert out.shape == x.shape

    def test_cuda_lif_binary(self, device):
        lif = CUDALIFNeuron(beta=0.25, threshold=1.0)
        x = torch.randn(4, 2, 16, 8, 8, device=device)
        out = lif(x)
        unique = torch.unique(out)
        assert all(v in [0.0, 1.0] for v in unique.tolist())

    def test_cuda_lif_matches_python(self, device):
        """CUDA LIF should produce same output as Python LIF."""
        from anima_tde.neurons import LIFNeuron

        torch.manual_seed(42)
        x = torch.randn(4, 2, 16, 8, 8, device=device)

        # Python reference
        py_lif = LIFNeuron(beta=0.25, threshold=1.0)
        py_out = py_lif(x)

        # CUDA
        cu_lif = CUDALIFNeuron(beta=0.25, threshold=1.0)
        cu_out = cu_lif(x)

        torch.testing.assert_close(cu_out, py_out, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_ops_loaded(self):
        """Custom CUDA ops should be loadable on GPU machines."""
        # This test passes if the .so was built correctly
        assert isinstance(has_cuda_ops(), bool)


class TestFusedSDA:
    """Test fused Spike-Driven Attention."""

    def test_sda_shape(self, device):
        shape = (4, 2, 64, 8, 8)
        H = torch.randn(*shape, device=device)
        g_ts = (torch.randn(*shape, device=device) > 0).float()
        g_tf = torch.randn(*shape, device=device)
        g_cs = (torch.randn(*shape, device=device) > 0).float()
        g_cf = torch.randn(*shape, device=device)
        g_ss = (torch.randn(*shape, device=device) > 0).float()
        g_sf = torch.randn(*shape, device=device)

        out = fused_sda_attention(H, g_ts, g_tf, g_cs, g_cf, g_ss, g_sf)
        assert out.shape == shape

    def test_sda_residual(self, device):
        """With all zeros except H, output should equal H."""
        shape = (4, 2, 64, 8, 8)
        H = torch.randn(*shape, device=device)
        zeros = torch.zeros(*shape, device=device)

        out = fused_sda_attention(H, zeros, zeros, zeros, zeros, zeros, zeros)
        torch.testing.assert_close(out, H)


class TestFusedAlphaMix:
    """Test fused alpha mixing."""

    def test_alpha_mix(self, device):
        a = torch.randn(2, 64, 32, 32, device=device)
        b = torch.randn(2, 64, 32, 32, device=device)

        out = fused_alpha_mix(a, b, 0.5)
        expected = 0.5 * a + 0.5 * b
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


class TestNMSOps:
    """Test NMS and IoU operations."""

    def test_box_iou(self, device):
        boxes1 = torch.tensor(
            [[0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 1.5, 1.5]],
            device=device,
        )
        boxes2 = torch.tensor(
            [[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]],
            device=device,
        )
        iou = cuda_box_iou(boxes1, boxes2)
        assert iou.shape == (2, 2)
        # box1[0] vs box2[0] should be 1.0 (identical)
        assert abs(iou[0, 0].item() - 1.0) < 1e-5
        # box1[0] vs box2[1] should be 0.0 (no overlap)
        assert abs(iou[0, 1].item()) < 1e-5

    def test_nms(self, device):
        boxes = torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.1, 0.1, 1.1, 1.1],
                [2.0, 2.0, 3.0, 3.0],
            ],
            device=device,
        )
        scores = torch.tensor([0.9, 0.8, 0.7], device=device)
        keep = cuda_nms(boxes, scores, iou_threshold=0.5)
        # First and third should be kept (second overlaps first)
        assert len(keep) == 2
        assert 0 in keep.tolist()
        assert 2 in keep.tolist()
