#!/usr/bin/env python3
"""CUDA-accelerated training entry point for TDE spiking object detector.

Uses fused CUDA kernels for:
- LIF neuron forward/backward (spiking_ops)
- SDA attention (accumulation-only fused kernel)
- Shared detection_ops (NMS, IoU)
- Shared fused_image_preprocess (normalize, augment)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_cu.py --config configs/paper.toml
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure src/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from anima_tde.backends.cuda.spiking_cuda import has_cuda_ops
from anima_tde.train import train
from anima_tde.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="TDE CUDA-Accelerated Training")
    parser.add_argument(
        "--config", type=str, default="configs/paper.toml",
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint for resuming",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Override max training steps (smoke test)",
    )
    args = parser.parse_args()

    # Report CUDA status
    print("=" * 60)
    print("TDE CUDA-Accelerated Training")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"VRAM: {total:.1f} GB")
    print(f"Custom CUDA ops: {has_cuda_ops()}")

    # Check shared CUDA extensions
    _check_shared_extensions()

    config = load_config(args.config)

    if args.max_steps is not None:
        config["training"]["epochs"] = 1
        print(f"[DEBUG] Max steps override: {args.max_steps}")

    print(f"[CONFIG] {args.config}")
    print(f"[MODEL] backbone={config['model']['backbone']}, "
          f"T={config['model']['timesteps']}, "
          f"variant={config['model']['tde_variant']}")
    print("=" * 60)

    train(config, resume=args.resume)


def _check_shared_extensions() -> None:
    """Report availability of shared CUDA extensions."""
    extensions = {
        "detection_ops": "/mnt/forge-data/shared_infra/cuda_extensions/detection_ops",
        "fused_image_preprocess": "/mnt/forge-data/shared_infra/cuda_extensions/fused_image_preprocess",
    }
    for name, path in extensions.items():
        so_path = os.path.join(path, f"{name}.cpython-311-x86_64-linux-gnu.so")
        status = "READY" if os.path.exists(so_path) else "NOT FOUND"
        print(f"Shared kernel {name}: {status}")


if __name__ == "__main__":
    main()
