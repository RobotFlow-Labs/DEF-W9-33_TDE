#!/usr/bin/env python3
"""Evaluation entry point for TDE.

Usage:
    python scripts/evaluate.py --config configs/paper.toml --checkpoint path/to/best.pth
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from anima_tde.evaluate import run_evaluation
from anima_tde.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="TDE Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/paper.toml",
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path for results",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_evaluation(config, args.checkpoint)

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"  mAP@50:     {results.get('mAP@50', 0):.4f}")
    print(f"  mAP@50:95:  {results.get('mAP@50:95', 0):.4f}")
    print(f"  Energy:     {results.get('energy_uj', 0):.2f} uJ")
    print(f"  Parameters: {results.get('num_parameters', 0) / 1e6:.2f}M")
    print(f"  Variant:    {results.get('tde_variant', 'unknown')}")
    print("=" * 60)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[SAVED] Results to {args.output}")


if __name__ == "__main__":
    main()
