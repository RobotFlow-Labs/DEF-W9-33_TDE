# PRD-04: Training Pipeline

> Status: TODO
> Module: 33_TDE
> Depends on: PRD-02, PRD-03

## Objective
Build the complete training pipeline: dataset loaders, training loop,
checkpointing, logging, early stopping, and resume capability.

## Components

### 1. Dataset Loaders (src/anima_tde/dataset.py)
- VOCDataset: loads PASCAL VOC 2007/2012 in YOLO format
- EvDETDataset: loads EvDET200K neuromorphic event frames
- Data augmentation: Mosaic, random flip, HSV jitter, multi-scale
- Temporal encoding: repeat static image T times (VOC) or stack event frames (EvDET)
- Train/val/test splits with saved indices

### 2. Training Loop (src/anima_tde/train.py)
- SGD optimizer, lr=0.01, momentum=0.937, weight_decay=5e-4
- Cosine annealing scheduler with 5% linear warmup
- Mixed precision (bf16 on CUDA)
- Gradient clipping max_norm=1.0
- Seed=42 for reproducibility

### 3. Checkpointing
- Save every 500 steps (top 2 by val_loss)
- best.pth always maintained
- Full state: model, optimizer, scheduler, epoch, step, metrics, config

### 4. Logging
- TensorBoard to /mnt/artifacts-datai/tensorboard/project_tde/
- JSON metrics to /mnt/artifacts-datai/logs/project_tde/
- Console: [Epoch N/total] train_loss=X val_loss=Y lr=Z

### 5. Early Stopping
- Patience: 20 epochs
- Min delta: 1e-4
- NaN detection with immediate halt

## Acceptance Criteria
- Training loop runs for 2 epochs on debug config without error
- Checkpoint save/load/resume cycle works
- Val loss logged every epoch
- TensorBoard events written
- nohup+disown compatible
