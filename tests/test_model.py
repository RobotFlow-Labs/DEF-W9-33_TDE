"""Tests for TDE model architecture."""

from __future__ import annotations

import torch

from anima_tde.model import (
    DetectionHead,
    SpikeDriverAttention,
    SpikeYOLOBackbone,
    SpikingEncoder,
    TCSAttention,
    TDEDetector,
    build_model,
)
from anima_tde.neurons import LIF0Neuron, LIF1Neuron, LIFNeuron


class TestLIFNeurons:
    """Test LIF neuron implementations."""

    def test_lif_output_shape(self):
        lif = LIFNeuron(beta=0.25, threshold=1.0)
        x = torch.randn(4, 2, 16, 8, 8)  # T=4, B=2, C=16, H=8, W=8
        out = lif(x)
        assert out.shape == x.shape

    def test_lif_binary_output(self):
        lif = LIFNeuron(beta=0.25, threshold=1.0)
        x = torch.randn(4, 2, 16, 8, 8)
        out = lif(x)
        unique = torch.unique(out)
        assert all(v in [0.0, 1.0] for v in unique.tolist())

    def test_lif_gradient_flows(self):
        lif = LIFNeuron(beta=0.25, threshold=1.0)
        x = torch.randn(4, 2, 16, 8, 8, requires_grad=True)
        out = lif(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_lif0_output_shape(self):
        lif0 = LIF0Neuron(beta=0.25, topk_percent=50)
        x = torch.randn(4, 2, 64)
        out = lif0(x)
        assert out.shape == x.shape

    def test_lif1_dual_output(self):
        lif1 = LIF1Neuron(beta=0.25, threshold=1.0)
        x = torch.randn(4, 2, 16)
        spike, membrane = lif1(x)
        assert spike.shape == x.shape
        assert membrane.shape == x.shape


class TestSpikingEncoder:
    """Test Spiking Encoder."""

    def test_forward_shape(self):
        se = SpikingEncoder(in_channels=3, out_channels=64, timesteps=4)
        img = torch.randn(2, 3, 32, 32)
        out = se(img)
        assert out.shape == (4, 2, 64, 32, 32)

    def test_single_channel_input(self):
        se = SpikingEncoder(in_channels=1, out_channels=32, timesteps=2)
        img = torch.randn(2, 1, 16, 16)
        out = se(img)
        assert out.shape == (2, 2, 32, 16, 16)


class TestAttention:
    """Test attention modules."""

    def test_sda_forward(self):
        sda = SpikeDriverAttention(channels=64, timesteps=4)
        x = torch.randn(4, 2, 64, 8, 8)
        out, temp_w = sda(x)
        assert out.shape == x.shape

    def test_tcsa_forward(self):
        tcsa = TCSAttention(channels=64, timesteps=4)
        x = torch.randn(4, 2, 64, 8, 8)
        out, temp_w = tcsa(x)
        assert out.shape == x.shape


class TestBackbone:
    """Test SpikeYOLO backbone."""

    def test_multi_scale_output(self):
        backbone = SpikeYOLOBackbone(in_channels=64, channels=(128, 256, 512))
        x = torch.randn(4, 2, 64, 32, 32)
        features = backbone(x)
        assert len(features) == 3
        assert features[0].shape == (4, 2, 128, 16, 16)
        assert features[1].shape == (4, 2, 256, 8, 8)
        assert features[2].shape == (4, 2, 512, 4, 4)


class TestDetectionHead:
    """Test detection head."""

    def test_output_shape(self):
        head = DetectionHead(in_channels=[128, 256, 512], num_classes=20)
        features = [
            torch.randn(4, 2, 128, 16, 16),
            torch.randn(4, 2, 256, 8, 8),
            torch.randn(4, 2, 512, 4, 4),
        ]
        outputs = head(features)
        assert len(outputs) == 3
        # Each output: (B, A*(5+20), H, W) = (2, 75, H, W)
        assert outputs[0].shape == (2, 75, 16, 16)
        assert outputs[1].shape == (2, 75, 8, 8)
        assert outputs[2].shape == (2, 75, 4, 4)


class TestTDEDetector:
    """Test full TDE detector."""

    def test_forward_sda(self):
        model = TDEDetector(
            num_classes=20,
            in_channels=3,
            timesteps=2,
            tde_variant="sda",
            se_out_channels=32,
            backbone_channels=(64, 128, 256),
            backbone_blocks=(1, 1, 1),
        )
        img = torch.randn(2, 3, 64, 64)
        outputs = model(img)
        assert len(outputs) == 3
        assert outputs[0].shape[0] == 2  # batch size

    def test_forward_tcsa(self):
        model = TDEDetector(
            num_classes=20,
            in_channels=3,
            timesteps=2,
            tde_variant="tcsa",
            se_out_channels=32,
            backbone_channels=(64, 128, 256),
            backbone_blocks=(1, 1, 1),
        )
        img = torch.randn(2, 3, 64, 64)
        outputs = model(img)
        assert len(outputs) == 3

    def test_gradient_flow(self):
        model = TDEDetector(
            num_classes=10,
            in_channels=1,
            timesteps=2,
            tde_variant="sda",
            se_out_channels=16,
            backbone_channels=(32, 64, 128),
            backbone_blocks=(1, 1, 1),
        )
        img = torch.randn(1, 1, 32, 32, requires_grad=True)
        outputs = model(img)
        loss = sum(o.sum() for o in outputs)
        loss.backward()
        assert img.grad is not None

    def test_parameter_count(self):
        model = TDEDetector(
            num_classes=20,
            timesteps=4,
            tde_variant="sda",
        )
        params = model.count_parameters()
        assert params > 0
        print(f"Parameter count: {params / 1e6:.2f}M")

    def test_energy_metrics(self):
        model = TDEDetector(tde_variant="sda")
        energy = model.count_energy()
        assert energy["mul_count"] == 0  # SDA has zero multiplications
        assert energy["ac_count"] > 0

    def test_build_from_config(self):
        config = {
            "model": {
                "num_classes": 20,
                "input_channels": 3,
                "timesteps": 2,
                "tde_variant": "sda",
                "lif": {"membrane_decay": 0.25},
                "se": {"out_channels": 32},
                "agm": {"reduction_ratio": 16},
            }
        }
        model = build_model(config)
        assert isinstance(model, TDEDetector)
