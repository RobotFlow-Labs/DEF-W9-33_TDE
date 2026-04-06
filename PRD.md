# PRD.md -- TDE Master Build Plan

> Module: 33_TDE (Wave 9 Defense)
> Paper: arXiv 2512.02447 -- Temporal Dynamics Enhancer for Directly Trained Spiking Object Detectors
> Last updated: 2026-04-05

## Build Plan

| PRD | Title | Status | Description |
|-----|-------|--------|-------------|
| PRD-01 | Foundation | [ ] TODO | Project scaffold, venv, pyproject.toml, configs, CI |
| PRD-02 | Core Model | [ ] TODO | LIF neurons, SE, AGM, SDA, SpikeYOLO backbone |
| PRD-03 | Loss Functions | [ ] TODO | Detection loss (CIoU + BCE cls + BCE obj), energy loss |
| PRD-04 | Training Pipeline | [ ] TODO | Dataset loaders, training loop, checkpointing, logging |
| PRD-05 | Evaluation | [ ] TODO | mAP@50, mAP@50-95, energy metrics, visualization |
| PRD-06 | Export Pipeline | [ ] TODO | ONNX, TensorRT fp16/fp32, safetensors |
| PRD-07 | Integration | [ ] TODO | Docker serve, ROS2 node, anima_module.yaml, HF push |

## Architecture Overview

```
Input Image (C x H x W)
    |
    v
[Spiking Encoder (SE)] -- generates T timesteps of diverse spike stimuli
    |                       alpha_t controls mix of current vs previous features
    v
[SNN Backbone] -- SpikeYOLO or EMS-YOLO (direct-trained SNN detector)
    |               uses LIF neurons with surrogate gradient
    v
[Attention Gating Module (AGM)] -- temporal + channel + spatial attention
    |                               updates alpha_t for next forward pass
    v
[Detection Head] -- YOLO-style multi-scale detection
    |
    v
Bounding Boxes + Class Predictions
```

## Key Design Decisions

1. **Backbone**: Implement SpikeYOLO as primary backbone (ECCV 2024, best results)
2. **Timesteps**: T=4 (paper default, good accuracy/efficiency tradeoff)
3. **Attention**: Implement both TCSA and SDA variants
4. **Neuron**: LIF with surrogate gradient (spikingjelly)
5. **Framework**: Custom implementation (not ultralytics fork) for clean code
6. **Training**: SGD, lr=0.01, 300 epochs, mosaic augmentation
