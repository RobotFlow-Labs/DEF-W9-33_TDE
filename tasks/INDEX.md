# Tasks INDEX -- TDE

> Module: 33_TDE
> Last updated: 2026-04-05

## PRD-01: Foundation
- [x] T01-01: Create pyproject.toml with hatchling, torch cu128, spikingjelly
- [x] T01-02: Create configs/paper.toml with paper hyperparameters
- [x] T01-03: Create configs/debug.toml for smoke testing
- [x] T01-04: Create src/anima_tde/__init__.py with version
- [x] T01-05: Create anima_module.yaml
- [ ] T01-06: Create .venv and run uv sync
- [ ] T01-07: Run ruff check on all files

## PRD-02: Core Model
- [x] T02-01: Implement LIF neuron with surrogate gradient (neurons.py)
- [x] T02-02: Implement LIF0 (top-k% firing) and LIF1 (dual output)
- [x] T02-03: Implement Spiking Encoder (SE) with alpha mixing
- [x] T02-04: Implement Temporal Attention branch
- [x] T02-05: Implement Channel Attention branch
- [x] T02-06: Implement Spatial Attention branch
- [x] T02-07: Implement AGM with TCSA variant
- [x] T02-08: Implement AGM with SDA variant (accumulation-only)
- [x] T02-09: Implement SpikeYOLO-style backbone with spiking residual blocks
- [x] T02-10: Implement multi-scale detection head
- [x] T02-11: Implement TDE wrapper (SE + backbone + AGM)
- [x] T02-12: Write test_model.py with forward pass + gradient test

## PRD-03: Loss Functions
- [x] T03-01: Implement CIoU loss
- [x] T03-02: Implement BCE objectness loss
- [x] T03-03: Implement BCE classification loss
- [x] T03-04: Implement combined detection loss with scale weights
- [x] T03-05: Implement spike rate regularization loss
- [x] T03-06: Implement label assignment (SimOTA-style)
- [ ] T03-07: Write test for loss backward pass

## PRD-04: Training Pipeline
- [x] T04-01: Implement VOCDataset with YOLO-format loading
- [x] T04-02: Implement EvDETDataset for neuromorphic events
- [x] T04-03: Implement mosaic data augmentation
- [x] T04-04: Implement temporal encoding (repeat for static, stack for events)
- [x] T04-05: Implement training loop with SGD + cosine scheduler
- [x] T04-06: Implement checkpoint manager (top-2, step-based)
- [x] T04-07: Implement early stopping
- [x] T04-08: Implement TensorBoard + JSON logging
- [x] T04-09: Implement resume from checkpoint
- [x] T04-10: Write scripts/train.py CLI entry point
- [x] T04-11: Write test_dataset.py

## PRD-05: Evaluation
- [x] T05-01: Implement mAP@50 computation
- [x] T05-02: Implement mAP@50:95 computation
- [x] T05-03: Implement energy metric calculation
- [ ] T05-04: Implement detection visualization
- [ ] T05-05: Implement spike activity heatmap
- [x] T05-06: Write scripts/evaluate.py CLI entry point
- [ ] T05-07: Implement TRAINING_REPORT.md generation

## PRD-06: Export Pipeline
- [ ] T06-01: Implement safetensors export
- [ ] T06-02: Implement ONNX export (unrolled T=4)
- [ ] T06-03: Implement TRT export via shared toolkit
- [ ] T06-04: Implement weight fidelity validation
- [ ] T06-05: Write scripts/export.py CLI entry point

## PRD-07: Integration
- [x] T07-01: Create Dockerfile.serve
- [x] T07-02: Create docker-compose.serve.yml
- [ ] T07-03: Implement serve.py (AnimaNode subclass)
- [ ] T07-04: Implement /predict endpoint
- [ ] T07-05: Test Docker build
- [ ] T07-06: Push to HuggingFace
