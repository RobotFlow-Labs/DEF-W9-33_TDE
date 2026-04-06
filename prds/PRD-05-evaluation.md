# PRD-05: Evaluation

> Status: TODO
> Module: 33_TDE
> Depends on: PRD-04

## Objective
Implement evaluation pipeline: mAP computation, energy metrics, visualization.

## Components

### 1. mAP Evaluation (src/anima_tde/evaluate.py)
- mAP@50 (VOC-style, per-class AP at IoU=0.50)
- mAP@50:95 (COCO-style, averaged over IoU 0.50:0.05:0.95)
- Per-class AP breakdown
- NMS post-processing (IoU threshold 0.45, confidence 0.001)

### 2. Energy Metrics
- Count MAC operations (multiply-accumulate)
- Separate MUL and AC counts
- Energy at 45nm: MUL=3.7pJ, AC=0.9pJ
- Compare TCSA vs SDA energy

### 3. Visualization
- Detection result overlay on images
- Spike activity heatmaps per timestep
- Attention weight visualization (temporal, channel, spatial)
- Training loss curves

### 4. Reporting
- Generate TRAINING_REPORT.md with metrics table
- Export metrics as JSON for automated comparison

## Target Metrics (paper reference)
| Dataset | Metric | SpikeYOLO+TDE-SDA |
|---------|--------|--------------------|
| VOC2007 | mAP@50 | 56.2% |
| VOC07+12 | mAP@50-95 | 57.7% |
| EvDET200K | mAP@50-95 | 47.6% |

## Acceptance Criteria
- mAP computation matches pycocotools reference on known predictions
- Energy calculation produces values in paper's range
- Visualization scripts generate valid images
- Report generation works end-to-end
