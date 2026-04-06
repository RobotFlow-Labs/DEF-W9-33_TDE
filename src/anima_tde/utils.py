"""Utility functions for TDE module."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch


def load_config(config_path: str) -> dict:
    """Load TOML configuration file.

    Args:
        config_path: Path to .toml config file.

    Returns:
        Parsed config as nested dict.
    """
    path = Path(config_path)
    if not path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    if sys.version_info >= (3, 11):
        import tomllib
    else:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

    with open(path, "rb") as f:
        return tomllib.load(f)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_gpu_memory(max_util: float = 0.80) -> None:
    """Check GPU memory utilization, raise if above cap."""
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_mem
        used = torch.cuda.memory_allocated(i)
        util = used / total
        if util > max_util:
            msg = (
                f"GPU {i} at {util * 100:.1f}% VRAM -- exceeds "
                f"{max_util * 100:.0f}% cap. Reduce batch_size."
            )
            raise RuntimeError(msg)


def get_project_name() -> str:
    """Return the ANIMA project name."""
    return "project_tde"


def get_artifacts_dir() -> Path:
    """Return the artifacts directory for this project."""
    return Path("/mnt/artifacts-datai")


def get_checkpoint_dir() -> Path:
    """Return checkpoint directory."""
    d = get_artifacts_dir() / "checkpoints" / get_project_name()
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_log_dir() -> Path:
    """Return log directory."""
    d = get_artifacts_dir() / "logs" / get_project_name()
    d.mkdir(parents=True, exist_ok=True)
    return d
