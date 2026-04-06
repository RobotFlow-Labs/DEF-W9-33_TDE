# PRD.md -- TDE Master Build Plan

> Module: 33_TDE (Wave 9 Defense)
> Paper: arXiv 2512.02447 -- Temporal Dynamics Enhancer for Directly Trained Spiking Object Detectors
> Last updated: 2026-04-06

## Build Plan

| PRD | Title | Status | Description |
|-----|-------|--------|-------------|
| PRD-01 | Foundation | [x] DONE | Project scaffold, venv, pyproject.toml, configs, CI |
| PRD-02 | Core Model | [x] DONE | LIF neurons, SE, AGM, SDA, SpikeYOLO backbone |
| PRD-03 | Loss Functions | [x] DONE | Detection loss (CIoU + BCE cls + BCE obj), energy loss |
| PRD-04 | Training Pipeline | [x] DONE | Dataset loaders, training loop, checkpointing, logging, CUDA kernels |
| PRD-05 | Evaluation | [x] DONE | mAP@50, mAP@50-95, energy metrics, NMS decoder |
| PRD-06 | Export Pipeline | [x] DONE | ONNX, TensorRT fp16/fp32, safetensors |
| PRD-07 | Integration | [ ] TODO | Docker serve, ROS2 node, HF push (blocked: needs training) |

## Architecture Overview

```
Input Image (C x H x W)
    |
    v
[Spiking Encoder (SE)] -- generates T timesteps of diverse spike stimuli
    |                       alpha_t controls mix of current vs previous features
    v
[SNN Backbone] -- SpikeYOLO (direct-trained SNN detector)
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

## CUDA Acceleration

| Kernel | Type | Location |
|--------|------|----------|
| fused_lif_forward | Custom (sm_89) | backends/cuda/kernels/spiking_ops.cu |
| fused_lif_backward | Custom (sm_89) | backends/cuda/kernels/spiking_ops.cu |
| fused_sda_attention | Custom (sm_89) | backends/cuda/kernels/spiking_ops.cu |
| fused_alpha_mix | Custom (sm_89) | backends/cuda/kernels/spiking_ops.cu |
| detection_ops (NMS, IoU) | Shared infra | /mnt/forge-data/shared_infra/cuda_extensions/ |
| fused_image_preprocess | Shared infra | /mnt/forge-data/shared_infra/cuda_extensions/ |

## Key Design Decisions

1. **Backbone**: SpikeYOLO as primary backbone (ECCV 2024, best results)
2. **Timesteps**: T=4 (paper default)
3. **Attention**: Both TCSA and SDA variants implemented
4. **Neuron**: LIF with atan surrogate gradient (custom CUDA kernel)
5. **Framework**: Custom implementation (not ultralytics fork)
6. **Training**: SGD, lr=0.01, 300 epochs, mosaic augmentation
7. **ONNX**: Legacy exporter (dynamo has issues with temporal loops)
