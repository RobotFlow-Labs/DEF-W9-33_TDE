"""Detection output decoder with NMS.

Converts raw YOLO-style multi-scale predictions into final
bounding boxes with confidence scores and class IDs.
"""

from __future__ import annotations

import torch

from anima_tde.backends.cuda.nms_cuda import cuda_nms


def decode_predictions(
    predictions: list[torch.Tensor],
    num_classes: int = 20,
    num_anchors: int = 3,
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.45,
) -> list[dict]:
    """Decode multi-scale YOLO predictions to boxes.

    Args:
        predictions: List of tensors, each (B, A*(5+C), H, W).
        num_classes: Number of object classes.
        num_anchors: Number of anchors per cell.
        conf_threshold: Minimum objectness * class confidence.
        nms_threshold: IoU threshold for NMS.

    Returns:
        List of dicts (one per batch), each with:
            boxes: (N, 4) x1,y1,x2,y2 in normalized coords
            scores: (N,) confidence scores
            class_ids: (N,) integer class IDs
    """
    batch_size = predictions[0].shape[0]
    device = predictions[0].device
    results = []

    for b in range(batch_size):
        all_boxes = []
        all_scores = []
        all_classes = []

        for pred in predictions:
            _, _, H, W = pred.shape
            p = pred[b]  # (A*(5+C), H, W)
            p = p.view(num_anchors, 5 + num_classes, H, W)
            p = p.permute(0, 2, 3, 1)  # (A, H, W, 5+C)

            # Grid
            gy, gx = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing="ij",
            )
            gx = gx.unsqueeze(0).expand(num_anchors, -1, -1)
            gy = gy.unsqueeze(0).expand(num_anchors, -1, -1)

            # Decode xy (sigmoid + grid offset, normalized)
            cx = (torch.sigmoid(p[..., 0]) + gx) / W
            cy = (torch.sigmoid(p[..., 1]) + gy) / H
            # Decode wh (exp)
            bw = torch.exp(p[..., 2].clamp(-10, 10)) / W
            bh = torch.exp(p[..., 3].clamp(-10, 10)) / H

            # Convert to x1y1x2y2
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (A, H, W, 4)

            # Objectness
            obj = torch.sigmoid(p[..., 4])  # (A, H, W)
            # Class scores
            cls = torch.sigmoid(p[..., 5:])  # (A, H, W, C)

            # Combined score = obj * cls
            scores, class_ids = (obj.unsqueeze(-1) * cls).max(dim=-1)

            # Flatten
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            class_ids = class_ids.reshape(-1)

            # Filter by confidence
            mask = scores > conf_threshold
            all_boxes.append(boxes[mask])
            all_scores.append(scores[mask])
            all_classes.append(class_ids[mask])

        if all_boxes:
            boxes_cat = torch.cat(all_boxes)
            scores_cat = torch.cat(all_scores)
            classes_cat = torch.cat(all_classes)
        else:
            boxes_cat = torch.zeros(0, 4, device=device)
            scores_cat = torch.zeros(0, device=device)
            classes_cat = torch.zeros(0, dtype=torch.long, device=device)

        # Per-class NMS
        if boxes_cat.shape[0] > 0:
            # Offset boxes by class for batched NMS
            offset = classes_cat.float() * 10.0  # separate classes
            offset_boxes = boxes_cat + offset.unsqueeze(-1)
            keep = cuda_nms(offset_boxes, scores_cat, nms_threshold)
            boxes_cat = boxes_cat[keep]
            scores_cat = scores_cat[keep]
            classes_cat = classes_cat[keep]

        results.append({
            "boxes": boxes_cat,
            "scores": scores_cat,
            "class_ids": classes_cat,
        })

    return results
