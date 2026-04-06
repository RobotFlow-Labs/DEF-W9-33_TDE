"""Backend auto-detection: CUDA > MLX > CPU."""

from __future__ import annotations

import importlib.util


def get_backend() -> str:
    """Detect available backend."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    if importlib.util.find_spec("mlx") is not None:
        return "mlx"

    return "cpu"


BACKEND = get_backend()
