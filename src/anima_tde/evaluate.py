"""Evaluation pipeline for TDE.

Implements:
- mAP@50 and mAP@50:95 computation
- Energy consumption metrics
- Per-class AP breakdown

Reference: arXiv 2512.02447, Section 4 (Experiments)
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from anima_tde.dataset import build_dataset, collate_fn
from anima_tde.model import TDEDetector, build_model


def compute_iou_matrix(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
) -> np.ndarray:
    """Compute IoU matrix between two sets of boxes (x1,y1,x2,y2).

    Args:
        boxes1: (N, 4) array.
        boxes2: (M, 4) array.

    Returns:
        (N, M) IoU matrix.
    """
    x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0:1].T)
    y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1:2].T)
    x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2:3].T)
    y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3:4].T)

    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-7)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute Average Precision using 11-point interpolation (VOC-style)."""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        prec_at_recall = precision[recall >= t]
        if prec_at_recall.size > 0:
            ap += prec_at_recall.max() / 11.0
    return ap


def compute_ap_all_point(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute AP using all-point interpolation (COCO-style)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Ensure precision is monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)


def evaluate_map(
    all_predictions: list[dict],
    all_targets: list[dict],
    iou_threshold: float = 0.5,
    num_classes: int = 20,
) -> dict[str, float]:
    """Compute mAP at a single IoU threshold.

    Args:
        all_predictions: List of dicts with keys:
            boxes: (N, 4) x1y1x2y2
            scores: (N,)
            classes: (N,) int
        all_targets: List of dicts with keys:
            boxes: (M, 4) x1y1x2y2
            classes: (M,) int
        iou_threshold: IoU threshold for positive match.
        num_classes: Number of object classes.

    Returns:
        Dict with 'mAP', 'AP_per_class'.
    """
    # Gather per-class predictions and targets
    class_preds: dict[int, list] = defaultdict(list)
    class_targets: dict[int, list] = defaultdict(list)

    for img_idx, (pred, tgt) in enumerate(
        zip(all_predictions, all_targets, strict=True)
    ):
        for i in range(len(pred["scores"])):
            cls = int(pred["classes"][i])
            class_preds[cls].append({
                "img_idx": img_idx,
                "score": float(pred["scores"][i]),
                "box": pred["boxes"][i],
            })
        for i in range(len(tgt["classes"])):
            cls = int(tgt["classes"][i])
            class_targets[cls].append({
                "img_idx": img_idx,
                "box": tgt["boxes"][i],
            })

    ap_per_class = {}

    for cls_id in range(num_classes):
        preds = class_preds.get(cls_id, [])
        targets = class_targets.get(cls_id, [])

        if len(targets) == 0:
            ap_per_class[cls_id] = 0.0
            continue

        # Sort by confidence
        preds.sort(key=lambda x: x["score"], reverse=True)

        # Track which targets have been matched
        matched = set()
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        # Group targets by image
        tgt_by_img: dict[int, list] = defaultdict(list)
        for i, t in enumerate(targets):
            tgt_by_img[t["img_idx"]].append((i, t["box"]))

        for pi, pred in enumerate(preds):
            img_idx = pred["img_idx"]
            pred_box = pred["box"]

            best_iou = 0.0
            best_ti = -1

            for ti, tgt_box in tgt_by_img.get(img_idx, []):
                iou = compute_iou_matrix(
                    np.array([pred_box]),
                    np.array([tgt_box]),
                )[0, 0]
                if iou > best_iou:
                    best_iou = iou
                    best_ti = ti

            if best_iou >= iou_threshold and best_ti not in matched:
                tp[pi] = 1
                matched.add(best_ti)
            else:
                fp[pi] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        recall = cum_tp / len(targets)
        precision = cum_tp / (cum_tp + cum_fp + 1e-7)

        ap_per_class[cls_id] = compute_ap_all_point(recall, precision)

    mean_ap = np.mean(list(ap_per_class.values())) if ap_per_class else 0.0

    return {
        "mAP": float(mean_ap),
        "AP_per_class": ap_per_class,
    }


def evaluate_map_coco(
    all_predictions: list[dict],
    all_targets: list[dict],
    num_classes: int = 20,
) -> dict[str, float]:
    """Compute mAP@50:95 (COCO-style).

    Averages AP across IoU thresholds [0.50, 0.55, ..., 0.95].
    """
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []

    for iou_thr in iou_thresholds:
        result = evaluate_map(
            all_predictions, all_targets, iou_thr, num_classes
        )
        aps.append(result["mAP"])

    map50 = aps[0]
    map50_95 = float(np.mean(aps))

    return {
        "mAP@50": map50,
        "mAP@50:95": map50_95,
        "AP_per_iou": {
            f"{t:.2f}": ap for t, ap in zip(iou_thresholds, aps, strict=True)
        },
    }


def compute_energy_metrics(
    model: TDEDetector,
    mul_pj: float = 3.7,
    ac_pj: float = 0.9,
) -> dict[str, float]:
    """Compute energy consumption metrics.

    Based on 45nm technology node at 32-bit precision.
    SDA variant has zero floating-point multiplications.

    Args:
        model: TDE model instance.
        mul_pj: Energy per multiply operation (picojoules).
        ac_pj: Energy per accumulate operation (picojoules).

    Returns:
        Dict with energy metrics.
    """
    return model.count_energy(mul_pj=mul_pj, ac_pj=ac_pj)


@torch.no_grad()
def run_evaluation(config: dict, checkpoint_path: str) -> dict:
    """Run full evaluation on test set.

    Args:
        config: Parsed TOML config.
        checkpoint_path: Path to model checkpoint.

    Returns:
        Dict with all evaluation metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load test data
    test_dataset = build_dataset(config, split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    num_classes = config["model"].get("num_classes", 20)

    all_preds = []
    all_tgts = []

    for batch in test_loader:
        images = batch["images"].to(device)
        targets = batch["targets"]

        preds = model(images)

        # Decode predictions (simplified -- real impl needs NMS)
        # For each scale, decode to boxes + scores + classes
        for scale_pred in preds:
            batch_size = scale_pred.shape[0]
            for _ in range(batch_size):
                # Placeholder decode -- real implementation in PRD-05
                all_preds.append({
                    "boxes": np.zeros((0, 4)),
                    "scores": np.zeros(0),
                    "classes": np.zeros(0, dtype=int),
                })

        # Decode targets
        for b in range(images.shape[0]):
            mask = targets[:, 0] == b
            tgt_boxes = targets[mask, 2:]  # cx, cy, w, h
            tgt_cls = targets[mask, 1].numpy().astype(int)
            # Convert to x1y1x2y2
            if len(tgt_boxes) > 0:
                x1 = (tgt_boxes[:, 0] - tgt_boxes[:, 2] / 2).numpy()
                y1 = (tgt_boxes[:, 1] - tgt_boxes[:, 3] / 2).numpy()
                x2 = (tgt_boxes[:, 0] + tgt_boxes[:, 2] / 2).numpy()
                y2 = (tgt_boxes[:, 1] + tgt_boxes[:, 3] / 2).numpy()
                boxes = np.stack([x1, y1, x2, y2], axis=1)
            else:
                boxes = np.zeros((0, 4))

            all_tgts.append({
                "boxes": boxes,
                "classes": tgt_cls,
            })

    # Compute metrics
    map_results = evaluate_map_coco(all_preds, all_tgts, num_classes)
    energy_results = compute_energy_metrics(model)

    results = {
        **map_results,
        **energy_results,
        "num_parameters": model.count_parameters(),
        "tde_variant": config["model"].get("tde_variant", "sda"),
    }

    print(f"[EVAL] mAP@50: {map_results['mAP@50']:.4f}")
    print(f"[EVAL] mAP@50:95: {map_results['mAP@50:95']:.4f}")
    print(f"[EVAL] Energy: {energy_results['energy_uj']:.2f} uJ")

    return results
