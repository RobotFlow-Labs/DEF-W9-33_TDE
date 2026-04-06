"""CUDA-accelerated NMS and IoU using shared detection_ops kernel.

Uses pre-compiled detection_ops from /mnt/forge-data/shared_infra/cuda_extensions/
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Try loading shared detection_ops
_det_ops = None
_shared_ext = Path("/mnt/forge-data/shared_infra/cuda_extensions/detection_ops")
if _shared_ext.exists():
    sys.path.insert(0, str(_shared_ext))
    try:
        import detection_ops

        _det_ops = detection_ops
    except ImportError:
        pass
    finally:
        sys.path.pop(0)


def has_detection_ops() -> bool:
    """Check if shared detection CUDA ops are available."""
    return _det_ops is not None


def cuda_box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """Compute 2D box IoU matrix using CUDA kernel.

    Args:
        boxes1: (N, 4) x1,y1,x2,y2
        boxes2: (M, 4) x1,y1,x2,y2

    Returns:
        (N, M) IoU matrix
    """
    if _det_ops is not None and boxes1.is_cuda:
        return _det_ops.fused_box_iou_2d(boxes1.contiguous(), boxes2.contiguous())
    # CPU fallback
    return _cpu_box_iou(boxes1, boxes2)


def _cpu_box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """CPU fallback for box IoU."""
    x1 = torch.max(boxes1[:, 0:1], boxes2[:, 0:1].T)
    y1 = torch.max(boxes1[:, 1:2], boxes2[:, 1:2].T)
    x2 = torch.min(boxes1[:, 2:3], boxes2[:, 2:3].T)
    y2 = torch.min(boxes1[:, 3:4], boxes2[:, 3:4].T)
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-7)


def cuda_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.45,
) -> torch.Tensor:
    """Non-Maximum Suppression using torchvision (CUDA-accelerated).

    Args:
        boxes: (N, 4) x1,y1,x2,y2
        scores: (N,) confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        Indices of kept boxes
    """
    try:
        from torchvision.ops import nms

        return nms(boxes, scores, iou_threshold)
    except ImportError:
        # Manual NMS fallback
        return _manual_nms(boxes, scores, iou_threshold)


def _manual_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Simple NMS fallback."""
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        remaining = order[1:]
        ious = cuda_box_iou(
            boxes[i : i + 1], boxes[remaining]
        ).squeeze(0)
        mask = ious <= iou_threshold
        order = remaining[mask]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
