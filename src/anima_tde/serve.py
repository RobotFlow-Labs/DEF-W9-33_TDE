"""TDE serving node -- AnimaNode subclass for Docker deployment.

Implements FastAPI inference endpoint and optional ROS2 node.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

try:
    from anima_serve.node import AnimaNode
except ImportError:
    # Fallback base class when anima_serve not installed
    class AnimaNode:
        """Stub base class for development without anima_serve."""

        def __init__(self, **kwargs):
            pass

        def setup_inference(self):
            raise NotImplementedError

        def process(self, input_data):
            raise NotImplementedError

        def get_status(self) -> dict:
            return {}

from anima_tde.model import build_model
from anima_tde.utils import load_config


class TDEDetectionNode(AnimaNode):
    """TDE spiking object detection serving node."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.device = None
        self.config = None
        self.conf_threshold = 0.25
        self.nms_threshold = 0.45

    def setup_inference(self):
        """Load model weights and configure backend."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load config
        config_path = Path(__file__).parent.parent.parent / "configs" / "paper.toml"
        if config_path.exists():
            self.config = load_config(str(config_path))
        else:
            self.config = {
                "model": {
                    "num_classes": 20,
                    "input_channels": 3,
                    "timesteps": 4,
                    "tde_variant": "sda",
                    "lif": {"membrane_decay": 0.25},
                    "se": {"out_channels": 64},
                    "agm": {"reduction_ratio": 16},
                },
            }

        # Build model
        self.model = build_model(self.config).to(self.device)
        self.model.eval()

        # Load weights if available
        weight_path = Path("/data/weights/best.pth")
        if weight_path.exists():
            ckpt = torch.load(
                weight_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(ckpt["model"])

    @torch.no_grad()
    def process(self, input_data: np.ndarray) -> dict:
        """Run detection on input image.

        Args:
            input_data: RGB image as numpy array (H, W, 3) uint8.

        Returns:
            Dict with boxes, scores, class_ids.
        """
        import cv2

        # Preprocess
        h, w = self.config["model"].get("input_height", 640), \
            self.config["model"].get("input_width", 640)
        img = cv2.resize(input_data, (w, h))
        img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        img_t = img_t.unsqueeze(0).to(self.device)

        # Inference
        t0 = time.time()
        _predictions = self.model(img_t)  # noqa: F841 — decode in production
        inference_time = time.time() - t0

        # Placeholder decode — real NMS decoding in production
        return {
            "boxes": [],
            "scores": [],
            "class_ids": [],
            "inference_time_ms": inference_time * 1000,
            "num_detections": 0,
        }

    def get_status(self) -> dict:
        """Module-specific status."""
        return {
            "model_loaded": self.model is not None,
            "device": str(self.device) if self.device else "none",
            "tde_variant": self.config["model"].get("tde_variant", "unknown")
            if self.config
            else "unknown",
            "timesteps": self.config["model"].get("timesteps", 4)
            if self.config
            else 4,
        }
