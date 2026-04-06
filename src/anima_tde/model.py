"""TDE model architecture.

Implements the full Temporal Dynamics Enhancer including:
- Spiking Encoder (SE): temporal diversity generation
- Attention Gating Module (AGM): multi-dimensional spike-driven attention
- SpikeYOLO backbone: spiking residual blocks + detection head

Reference: arXiv 2512.02447
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from anima_tde.neurons import LIF0Neuron, LIF1Neuron, LIFNeuron

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBN(nn.Module):
    """Conv2d + BatchNorm2d (no activation -- activation is spiking)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class SpikingResBlock(nn.Module):
    """Spiking residual block with LIF neurons.

    Conv-BN-LIF-Conv-BN + residual, followed by LIF.
    """

    def __init__(self, channels: int, beta: float = 0.25) -> None:
        super().__init__()
        self.conv1 = ConvBN(channels, channels, 3)
        self.conv2 = ConvBN(channels, channels, 3)
        self.lif = LIFNeuron(beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, B, C, H, W)."""
        T, B = x.shape[:2]
        out_frames = []
        for t in range(T):
            h = self.conv1(x[t])
            h = F.relu(h)  # pre-LIF activation approx
            h = self.conv2(h)
            h = h + x[t]  # residual
            out_frames.append(h)
        stacked = torch.stack(out_frames, dim=0)
        return self.lif(stacked)


class DownBlock(nn.Module):
    """Downsampling block: stride-2 conv + spiking residual blocks."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_blocks: int = 1,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.down = ConvBN(in_ch, out_ch, 3, stride=2)
        self.lif_down = LIFNeuron(beta=beta)
        self.blocks = nn.ModuleList(
            [SpikingResBlock(out_ch, beta=beta) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, B, C, H, W) -> (T, B, out_ch, H/2, W/2)."""
        T = x.shape[0]
        down_frames = []
        for t in range(T):
            down_frames.append(self.down(x[t]))
        h = self.lif_down(torch.stack(down_frames, dim=0))
        for block in self.blocks:
            h = block(h)
        return h


# ---------------------------------------------------------------------------
# Spiking Encoder (SE)
# ---------------------------------------------------------------------------

class SpikingEncoder(nn.Module):
    """Spiking Encoder: generates diverse input stimuli across timesteps.

    At t=0: A_t = ConvBN(I)
    At t>0: A_t = alpha_t * ConvBN(I) + (1 - alpha_t) * Conv_k(A_{t-1})
    S = LIF(A)

    Args:
        in_channels: Input image channels (3 for RGB, 1 for events).
        out_channels: Output feature channels.
        kernel_size: Kernel size for temporal conv.
        timesteps: Number of timesteps T.
        beta: LIF membrane decay.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        kernel_size: int = 3,
        timesteps: int = 4,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.input_conv = ConvBN(in_channels, out_channels, 3)
        self.temporal_conv = ConvBN(out_channels, out_channels, kernel_size)
        self.lif = LIFNeuron(beta=beta)

        # Learnable preference coefficients (initialized to 0.5)
        self.alpha = nn.Parameter(torch.full((timesteps,), 0.5))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Encode static image into temporal spike stream.

        Args:
            image: (B, C, H, W) input image.

        Returns:
            Spike stream (T, B, out_ch, H, W).
        """
        input_feat = self.input_conv(image)  # (B, out_ch, H, W)

        features = []
        prev_feat = None

        for t in range(self.timesteps):
            alpha_t = torch.sigmoid(self.alpha[t])

            if t == 0:
                feat = input_feat
            else:
                temporal_feat = self.temporal_conv(prev_feat)
                feat = alpha_t * input_feat + (1 - alpha_t) * temporal_feat

            features.append(feat)
            prev_feat = feat

        stacked = torch.stack(features, dim=0)  # (T, B, C, H, W)
        return self.lif(stacked)

    def update_alpha(self, attention_weights: torch.Tensor) -> None:
        """Update preference coefficients from AGM attention weights.

        Algorithm 1 from paper:
            alpha_hat = batch_mean(temporal_attention_float)
            alpha = 0.5 * (alpha_bar + alpha_hat)
        """
        with torch.no_grad():
            alpha_hat = attention_weights.mean(dim=0)  # Average over batch
            # Temporal smoothing
            self.alpha.data = 0.5 * (self.alpha.data + alpha_hat[: self.timesteps])


# ---------------------------------------------------------------------------
# Attention Gating Module (AGM) with SDA
# ---------------------------------------------------------------------------


def _global_max_pool2d(x: torch.Tensor) -> torch.Tensor:
    """ONNX-compatible global max pool: (B, C, H, W) -> (B, C, 1, 1)."""
    return x.flatten(2).max(dim=2).values.unsqueeze(-1).unsqueeze(-1)


class TemporalAttention(nn.Module):
    """Temporal attention branch using LIF0 + LIF1."""

    def __init__(self, channels: int, timesteps: int, beta: float = 0.25) -> None:
        super().__init__()
        # Global max pool is done via _global_max_pool2d (ONNX-compatible)
        self.lif0 = LIF0Neuron(beta=beta, topk_percent=50)
        self.fc = nn.Linear(channels, timesteps)
        self.lif1 = LIF1Neuron(beta=beta)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (T, B, C, H, W) -> spike (T, B, 1, 1, 1), float (T, B, 1, 1, 1)."""
        T, B, C, H, W = x.shape
        pooled = []
        for t in range(T):
            pooled.append(_global_max_pool2d(x[t]).view(B, C))
        pooled = torch.stack(pooled, dim=0)  # (T, B, C)

        h = self.lif0(pooled)  # (T, B, C)

        # FC across channels -> temporal weights
        out_frames = []
        for t in range(T):
            out_frames.append(self.fc(h[t]))  # (B, T)
        out = torch.stack(out_frames, dim=0)  # (T, B, T)

        spike, membrane = self.lif1(out)  # each (T, B, T)

        # Reshape for broadcasting: (T, B, 1, 1, 1)
        spike = spike[:, :, 0:1].unsqueeze(-1).unsqueeze(-1)
        membrane = membrane[:, :, 0:1].unsqueeze(-1).unsqueeze(-1)
        return spike, membrane


class ChannelAttention(nn.Module):
    """Channel attention branch using LIF0 + LIF1."""

    def __init__(self, channels: int, reduction: int = 16, beta: float = 0.25) -> None:
        super().__init__()
        # Global max pool is done via _global_max_pool2d (ONNX-compatible)
        self.lif0 = LIF0Neuron(beta=beta, topk_percent=50)
        mid = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)
        self.lif1 = LIF1Neuron(beta=beta)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (T, B, C, H, W) -> spike (T, B, C, 1, 1), float (T, B, C, 1, 1)."""
        T, B, C, H, W = x.shape
        pooled = []
        for t in range(T):
            pooled.append(_global_max_pool2d(x[t]).view(B, C))
        pooled = torch.stack(pooled, dim=0)  # (T, B, C)

        h = self.lif0(pooled)

        out_frames = []
        for t in range(T):
            z = F.relu(self.fc1(h[t]))
            z = self.fc2(z)
            out_frames.append(z)
        out = torch.stack(out_frames, dim=0)  # (T, B, C)

        spike, membrane = self.lif1(out)
        spike = spike.unsqueeze(-1).unsqueeze(-1)
        membrane = membrane.unsqueeze(-1).unsqueeze(-1)
        return spike, membrane


class SpatialAttention(nn.Module):
    """Spatial attention branch using LIF0 + LIF1."""

    def __init__(self, beta: float = 0.25) -> None:
        super().__init__()
        self.lif0 = LIF0Neuron(beta=beta, topk_percent=50)
        self.conv = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        self.lif1 = LIF1Neuron(beta=beta)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (T, B, C, H, W) -> spike (T, B, 1, H, W), float (T, B, 1, H, W)."""
        T, B, C, H, W = x.shape

        # Max pool across channels
        pooled = x.max(dim=2, keepdim=True).values  # (T, B, 1, H, W)
        h = self.lif0(pooled.view(T, B, H * W)).view(T, B, 1, H, W)

        out_frames = []
        for t in range(T):
            out_frames.append(self.conv(h[t]))
        out = torch.stack(out_frames, dim=0)

        spike, membrane = self.lif1(out)
        return spike, membrane


class SpikeDriverAttention(nn.Module):
    """Spike-Driven Attention (SDA) -- accumulation-only attention.

    Combines temporal, channel, and spatial attention using only
    accumulation operations (no multiplications between float tensors).

    H_Att = g_temp_spike * g_ch_float * g_spa_spike
          + g_ch_spike * g_spa_float * g_temp_spike
          + g_spa_spike * g_temp_float * g_ch_spike
          + H

    Because spike tensors are binary {0,1}, multiplying spike * float
    is just masking (accumulation), not floating-point multiplication.
    """

    def __init__(
        self,
        channels: int,
        timesteps: int = 4,
        reduction: int = 16,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.temporal = TemporalAttention(channels, timesteps, beta)
        self.channel = ChannelAttention(channels, reduction, beta)
        self.spatial = SpatialAttention(beta)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply SDA attention.

        Args:
            x: (T, B, C, H, W) hidden states.

        Returns:
            Tuple of (attended_features, temporal_float_weights).
            temporal_float_weights used for alpha update.
        """
        g_temp_s, g_temp_f = self.temporal(x)
        g_ch_s, g_ch_f = self.channel(x)
        g_spa_s, g_spa_f = self.spatial(x)

        # SDA: binary spike * float = accumulation only
        attended = (
            g_temp_s * g_ch_f * g_spa_s
            + g_ch_s * g_spa_f * g_temp_s
            + g_spa_s * g_temp_f * g_ch_s
            + x
        )

        return attended, g_temp_f


class TCSAttention(nn.Module):
    """Traditional Temporal-Channel-Spatial Attention (TCSA) baseline.

    Standard multiply-based attention for comparison with SDA.
    """

    def __init__(
        self,
        channels: int,
        timesteps: int = 4,
        reduction: int = 16,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.temporal = TemporalAttention(channels, timesteps, beta)
        self.channel = ChannelAttention(channels, reduction, beta)
        self.spatial = SpatialAttention(beta)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard multiply attention."""
        g_temp_s, g_temp_f = self.temporal(x)
        g_ch_s, g_ch_f = self.channel(x)
        g_spa_s, g_spa_f = self.spatial(x)

        # TCSA uses float * float multiplications
        attended = (g_temp_f * g_ch_f * g_spa_f) * x + x

        return attended, g_temp_f


# ---------------------------------------------------------------------------
# SpikeYOLO-style Backbone
# ---------------------------------------------------------------------------

class SpikeYOLOBackbone(nn.Module):
    """Simplified SpikeYOLO backbone with multi-scale output.

    Produces feature maps at 3 scales for detection head.
    Architecture follows SpikeYOLO (arXiv 2407.20708) with
    spiking residual blocks at each stage.
    """

    def __init__(
        self,
        in_channels: int = 64,
        channels: tuple[int, ...] = (128, 256, 512),
        num_blocks: tuple[int, ...] = (2, 2, 2),
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        assert len(channels) == len(num_blocks) == 3

        self.stage1 = DownBlock(in_channels, channels[0], num_blocks[0], beta)
        self.stage2 = DownBlock(channels[0], channels[1], num_blocks[1], beta)
        self.stage3 = DownBlock(channels[1], channels[2], num_blocks[2], beta)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """x: (T, B, in_ch, H, W) -> list of 3 feature maps at different scales."""
        p3 = self.stage1(x)   # (T, B, 128, H/2, W/2)
        p4 = self.stage2(p3)  # (T, B, 256, H/4, W/4)
        p5 = self.stage3(p4)  # (T, B, 512, H/8, W/8)
        return [p3, p4, p5]


# ---------------------------------------------------------------------------
# Detection Head
# ---------------------------------------------------------------------------

class DetectionHead(nn.Module):
    """YOLO-style detection head for spiking features.

    For each scale, predicts (num_anchors * (5 + num_classes)) channels.
    5 = (x, y, w, h, objectness).

    Temporal features are averaged across timesteps before prediction.
    """

    def __init__(
        self,
        in_channels: list[int],
        num_classes: int = 20,
        num_anchors: int = 3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        out_ch = num_anchors * (5 + num_classes)

        self.heads = nn.ModuleList([
            nn.Sequential(
                ConvBN(ch, ch, 3),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, out_ch, 1),
            )
            for ch in in_channels
        ])

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """features: list of (T, B, C, H, W) -> list of (B, A*(5+cls), H, W)."""
        outputs = []
        for feat, head in zip(features, self.heads, strict=True):
            # Average spike features across timesteps
            avg = feat.mean(dim=0)  # (B, C, H, W)
            outputs.append(head(avg))
        return outputs


# ---------------------------------------------------------------------------
# Full TDE Model
# ---------------------------------------------------------------------------

class TDEDetector(nn.Module):
    """Full TDE-enhanced spiking object detector.

    Pipeline:
        Image -> Spiking Encoder (SE) -> SNN Backbone -> AGM -> Detection Head

    Args:
        num_classes: Number of detection classes.
        in_channels: Input image channels (3=RGB, 1=events).
        timesteps: Number of SNN timesteps T.
        tde_variant: "sda" for Spike-Driven Attention, "tcsa" for standard.
        se_out_channels: Spiking Encoder output channels.
        backbone_channels: Channel dims for backbone stages.
        backbone_blocks: Number of residual blocks per stage.
        beta: LIF membrane decay factor.
        agm_reduction: Channel attention reduction ratio.
    """

    def __init__(
        self,
        num_classes: int = 20,
        in_channels: int = 3,
        timesteps: int = 4,
        tde_variant: str = "sda",
        se_out_channels: int = 64,
        backbone_channels: tuple[int, ...] = (128, 256, 512),
        backbone_blocks: tuple[int, ...] = (2, 2, 2),
        beta: float = 0.25,
        agm_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.tde_variant = tde_variant

        # Spiking Encoder
        self.encoder = SpikingEncoder(
            in_channels=in_channels,
            out_channels=se_out_channels,
            kernel_size=3,
            timesteps=timesteps,
            beta=beta,
        )

        # SNN Backbone
        self.backbone = SpikeYOLOBackbone(
            in_channels=se_out_channels,
            channels=backbone_channels,
            num_blocks=backbone_blocks,
            beta=beta,
        )

        # AGM for each backbone scale
        agm_cls = SpikeDriverAttention if tde_variant == "sda" else TCSAttention
        self.agm_modules = nn.ModuleList([
            agm_cls(
                channels=ch,
                timesteps=timesteps,
                reduction=agm_reduction,
                beta=beta,
            )
            for ch in backbone_channels
        ])

        # Detection head
        self.head = DetectionHead(
            in_channels=list(backbone_channels),
            num_classes=num_classes,
        )

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            image: (B, C, H, W) input image (or event frame).

        Returns:
            List of detection tensors, one per scale.
            Each: (B, num_anchors*(5+num_classes), H_i, W_i).
        """
        # Spiking Encoder: image -> spike stream
        spikes = self.encoder(image)  # (T, B, se_ch, H, W)

        # Backbone: extract multi-scale features
        features = self.backbone(spikes)  # list of (T, B, C, H, W)

        # AGM: apply attention at each scale
        attended = []
        for feat, agm in zip(features, self.agm_modules, strict=True):
            att_feat, _temp_weights = agm(feat)
            attended.append(att_feat)

        # Detection head
        detections = self.head(attended)
        return detections

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_energy(
        self,
        input_shape: tuple[int, ...] = (1, 3, 640, 640),
        mul_pj: float = 3.7,
        ac_pj: float = 0.9,
    ) -> dict[str, float]:
        """Estimate energy consumption (simplified).

        Returns dict with mul_count, ac_count, energy_uj.
        """
        # Simplified: count operations based on parameter shapes
        # Real energy counting requires profiling spike rates
        total_params = self.count_parameters()
        T = self.timesteps

        if self.tde_variant == "sda":
            mul_count = 0  # SDA has zero float multiplications
            ac_count = total_params * T  # approximate
        else:
            mul_count = total_params * T  # TCSA uses multiplications
            ac_count = total_params * T

        energy_uj = (mul_count * mul_pj + ac_count * ac_pj) / 1e6
        return {
            "mul_count": mul_count,
            "ac_count": ac_count,
            "energy_uj": energy_uj,
        }


def build_model(config: dict) -> TDEDetector:
    """Build TDE model from config dict.

    Args:
        config: Parsed TOML config with [model] section.

    Returns:
        TDEDetector instance.
    """
    model_cfg = config.get("model", {})
    lif_cfg = model_cfg.get("lif", {})
    se_cfg = model_cfg.get("se", {})
    agm_cfg = model_cfg.get("agm", {})

    return TDEDetector(
        num_classes=model_cfg.get("num_classes", 20),
        in_channels=model_cfg.get("input_channels", 3),
        timesteps=model_cfg.get("timesteps", 4),
        tde_variant=model_cfg.get("tde_variant", "sda"),
        se_out_channels=se_cfg.get("out_channels", 64),
        backbone_channels=(128, 256, 512),
        backbone_blocks=(2, 2, 2),
        beta=lif_cfg.get("membrane_decay", 0.25),
        agm_reduction=agm_cfg.get("reduction_ratio", 16),
    )
