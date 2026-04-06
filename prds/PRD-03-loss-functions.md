# PRD-03: Loss Functions

> Status: TODO
> Module: 33_TDE
> Depends on: PRD-02

## Objective
Implement all loss functions for training the spiking object detector.

## Components

### 1. Detection Loss (src/anima_tde/losses.py)
YOLO-style multi-component loss:
- **Box regression**: CIoU loss (Complete IoU)
- **Objectness**: BCE with logits
- **Classification**: BCE with logits (multi-label)
- **Total**: L = lambda_box * L_box + lambda_obj * L_obj + lambda_cls * L_cls

### 2. Loss Weights (from YOLO defaults)
- lambda_box: 0.05
- lambda_obj: 1.0 (scaled per detection scale)
- lambda_cls: 0.5

### 3. Label Assignment
- SimOTA or ATSS-style assignment (following SpikeYOLO)
- Multi-scale anchor matching
- IoU-based positive/negative assignment

### 4. Energy Loss (optional regularizer)
- Spike rate regularization to encourage sparse firing
- L_spike = mean(spike_rate) -- penalizes excessive firing

## Acceptance Criteria
- Loss computation runs without NaN on random inputs
- Gradient flows through all loss components
- CIoU loss matches reference implementation
- Loss decreases on a trivial overfitting test (1 batch, 100 steps)
