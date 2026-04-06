#!/usr/bin/env python3
"""Training entry point for TDE spiking object detector.

Usage:
    python scripts/train.py --config configs/paper.toml
    python scripts/train.py --config configs/debug.toml --resume path/to/ckpt.pth

Always run from project root with venv activated:
    cd /mnt/forge-data/modules/05_wave9/33_TDE
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/paper.toml
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure src/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from anima_tde.train import train
from anima_tde.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="TDE Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/paper.toml",
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max training steps (for smoke test)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Override epochs for smoke test
    if args.max_steps is not None:
        config["training"]["epochs"] = 1
        print(f"[DEBUG] Max steps override: {args.max_steps}")

    print(f"[CONFIG] {args.config}")
    print(f"[MODEL] backbone={config['model']['backbone']}, "
          f"T={config['model']['timesteps']}, "
          f"variant={config['model']['tde_variant']}")

    train(config, resume=args.resume)


if __name__ == "__main__":
    main()
