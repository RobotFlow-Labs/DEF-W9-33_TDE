"""Loss functions for TDE spiking object detector.

Implements YOLO-style detection loss with:
- CIoU box regression loss
- BCE objectness loss
- BCE classification loss
- Optional spike rate regularization

Reference: arXiv 2512.02447 + SpikeYOLO detection loss
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def box_ciou(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Complete IoU loss between predicted and target boxes.

    Boxes in (x1, y1, x2, y2) format.

    Args:
        pred: Predicted boxes (N, 4).
        target: Target boxes (N, 4).
        eps: Small value for numerical stability.

    Returns:
        CIoU loss values (N,).
    """
    # Intersection
    inter_x1 = torch.max(pred[:, 0], target[:, 0])
    inter_y1 = torch.max(pred[:, 1], target[:, 1])
    inter_x2 = torch.min(pred[:, 2], target[:, 2])
    inter_y2 = torch.min(pred[:, 3], target[:, 3])
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Union
    area_pred = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_target = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = area_pred + area_target - inter_area + eps

    iou = inter_area / union

    # Enclosing box
    enc_x1 = torch.min(pred[:, 0], target[:, 0])
    enc_y1 = torch.min(pred[:, 1], target[:, 1])
    enc_x2 = torch.max(pred[:, 2], target[:, 2])
    enc_y2 = torch.max(pred[:, 3], target[:, 3])

    # Diagonal distance of enclosing box
    c2 = (enc_x2 - enc_x1).pow(2) + (enc_y2 - enc_y1).pow(2) + eps

    # Center distance
    pred_cx = (pred[:, 0] + pred[:, 2]) / 2
    pred_cy = (pred[:, 1] + pred[:, 3]) / 2
    target_cx = (target[:, 0] + target[:, 2]) / 2
    target_cy = (target[:, 1] + target[:, 3]) / 2
    rho2 = (pred_cx - target_cx).pow(2) + (pred_cy - target_cy).pow(2)

    # Aspect ratio penalty
    pred_w = (pred[:, 2] - pred[:, 0]).clamp(min=eps)
    pred_h = (pred[:, 3] - pred[:, 1]).clamp(min=eps)
    target_w = (target[:, 2] - target[:, 0]).clamp(min=eps)
    target_h = (target[:, 3] - target[:, 1]).clamp(min=eps)

    v = (4 / (math.pi**2)) * (
        torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)
    ).pow(2)

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - rho2 / c2 - alpha * v
    return 1 - ciou  # CIoU loss (1 - CIoU for minimization)


class DetectionLoss(nn.Module):
    """Combined detection loss for spiking YOLO detector.

    L = lambda_box * L_ciou + lambda_obj * L_obj + lambda_cls * L_cls
      + lambda_spike * L_spike_reg

    Args:
        num_classes: Number of object classes.
        box_weight: Weight for CIoU box loss.
        obj_weight: Weight for objectness BCE loss.
        cls_weight: Weight for classification BCE loss.
        spike_reg_weight: Weight for spike rate regularization.
    """

    def __init__(
        self,
        num_classes: int = 20,
        box_weight: float = 0.05,
        obj_weight: float = 1.0,
        cls_weight: float = 0.5,
        spike_reg_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.spike_reg_weight = spike_reg_weight

        self.bce_obj = nn.BCEWithLogitsLoss(reduction="mean")
        self.bce_cls = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        predictions: list[torch.Tensor],
        targets: torch.Tensor,
        spike_rates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute detection loss.

        Args:
            predictions: List of detection tensors per scale,
                each (B, A*(5+C), H, W).
            targets: Ground truth boxes (N, 6) where columns are
                (batch_idx, class_id, cx, cy, w, h) in normalized coords.
            spike_rates: Optional tensor of average spike rates for
                regularization.

        Returns:
            Dict with 'total', 'box', 'obj', 'cls', 'spike_reg' losses.
        """
        device = predictions[0].device
        loss_box = torch.tensor(0.0, device=device)
        loss_obj = torch.tensor(0.0, device=device)
        loss_cls = torch.tensor(0.0, device=device)

        num_anchors = 3
        num_targets = 0

        for _scale_idx, pred in enumerate(predictions):
            B, _, H, W = pred.shape
            pred = pred.view(B, num_anchors, 5 + self.num_classes, H, W)
            pred = pred.permute(0, 1, 3, 4, 2)  # (B, A, H, W, 5+C)

            # Objectness target: all zeros initially
            obj_target = torch.zeros(B, num_anchors, H, W, device=device)

            # Extract predictions
            pred_obj = pred[..., 4]  # (B, A, H, W)

            if targets is not None and targets.shape[0] > 0:
                # Simple assignment: map targets to grid cells
                for ti in range(targets.shape[0]):
                    b_idx = int(targets[ti, 0])
                    cls_id = int(targets[ti, 1])
                    cx = targets[ti, 2] * W
                    cy = targets[ti, 3] * H
                    tw = targets[ti, 4] * W
                    th = targets[ti, 5] * H

                    gi, gj = int(cx.clamp(0, W - 1)), int(cy.clamp(0, H - 1))

                    for ai in range(num_anchors):
                        # Box loss
                        pred_box = pred[b_idx, ai, gj, gi, :4]
                        # Convert pred to x1y1x2y2
                        px1 = (torch.sigmoid(pred_box[0]) + gi - tw / 2) / W
                        py1 = (torch.sigmoid(pred_box[1]) + gj - th / 2) / H
                        px2 = (torch.sigmoid(pred_box[0]) + gi + tw / 2) / W
                        py2 = (torch.sigmoid(pred_box[1]) + gj + th / 2) / H
                        pred_xyxy = torch.stack([px1, py1, px2, py2]).unsqueeze(0)

                        tx1 = targets[ti, 2] - targets[ti, 4] / 2
                        ty1 = targets[ti, 3] - targets[ti, 5] / 2
                        tx2 = targets[ti, 2] + targets[ti, 4] / 2
                        ty2 = targets[ti, 3] + targets[ti, 5] / 2
                        tgt_xyxy = torch.stack([tx1, ty1, tx2, ty2]).unsqueeze(0)

                        loss_box = loss_box + box_ciou(pred_xyxy, tgt_xyxy).mean()

                        # Objectness: positive sample
                        obj_target[b_idx, ai, gj, gi] = 1.0

                        # Classification loss
                        cls_target = torch.zeros(
                            self.num_classes, device=device
                        )
                        cls_target[cls_id] = 1.0
                        pred_cls = pred[b_idx, ai, gj, gi, 5:]
                        loss_cls = loss_cls + self.bce_cls(
                            pred_cls, cls_target
                        )

                        num_targets += 1

            # Objectness loss (all positions)
            loss_obj = loss_obj + self.bce_obj(pred_obj, obj_target)

        # Normalize
        num_targets = max(num_targets, 1)
        loss_box = loss_box / num_targets
        loss_cls = loss_cls / num_targets

        # Spike rate regularization
        loss_spike = torch.tensor(0.0, device=device)
        if spike_rates is not None and self.spike_reg_weight > 0:
            loss_spike = spike_rates.mean()

        total = (
            self.box_weight * loss_box
            + self.obj_weight * loss_obj
            + self.cls_weight * loss_cls
            + self.spike_reg_weight * loss_spike
        )

        return {
            "total": total,
            "box": loss_box.detach(),
            "obj": loss_obj.detach(),
            "cls": loss_cls.detach(),
            "spike_reg": loss_spike.detach(),
        }


def build_loss(config: dict) -> DetectionLoss:
    """Build loss from config dict."""
    loss_cfg = config.get("loss", {})
    model_cfg = config.get("model", {})
    return DetectionLoss(
        num_classes=model_cfg.get("num_classes", 20),
        box_weight=loss_cfg.get("box_weight", 0.05),
        obj_weight=loss_cfg.get("obj_weight", 1.0),
        cls_weight=loss_cfg.get("cls_weight", 0.5),
        spike_reg_weight=loss_cfg.get("spike_reg_weight", 0.0),
    )
