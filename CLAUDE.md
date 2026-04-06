# TDE -- Temporal Dynamics Enhancer for Directly Trained Spiking Object Detectors

> Paper: arXiv 2512.02447
> Authors: Fan Luo, Zeyu Gao, Xinhao Luo, Kai Zhao, Yanfeng Lu
> Code: https://github.com/Mortal825/TDE
> Module: 33_TDE (Wave 9 Defense)

## Paper Summary

TDE is a plug-in module that strengthens temporal dynamics in directly-trained
Spiking Neural Networks (SNNs) for object detection. SNNs are energy-efficient
alternatives to ANNs that use binary spike-based computation. TDE addresses the
limitation that existing SNN detectors underutilize temporal information.

TDE has two components:
1. **Spiking Encoder (SE)** -- generates diverse input stimuli across timesteps
   by mixing the current input with convolved features from the previous step,
   controlled by a learnable preference coefficient alpha_t.
2. **Attention Gating Module (AGM)** -- applies multi-dimensional attention
   (temporal, channel, spatial) to guide the SE generation via inter-temporal
   dependencies. The attention weights update the preference coefficient.

To avoid expensive multiply operations in the AGM, the authors propose
**Spike-Driven Attention (SDA)** which replaces multiplications with
accumulation-only operations, consuming only 0.240x the energy of conventional
attention.

## Architecture

### Spiking Encoder (SE)
- Input: image I in R^(C_in x H x W)
- Output: spike stream S in {0,1}^(T x C_out x H x W)
- At t=0: A_t = f_0(I) (initial Conv-BN)
- At t>0: A_t = alpha_t * I + (1 - alpha_t) * Conv_k(A_{t-1})
- S = LIF(A) -- Leaky Integrate-and-Fire neuron converts to spikes

### Attention Gating Module (AGM)
- Input: hidden states H^n in R^(T x C x H x W)
- Temporal attention via MaxPool -> LIF0 -> Conv -> LIF1
- Channel attention via MaxPool -> LIF0 -> FC -> LIF1
- Spatial attention via MaxPool -> LIF0 -> Conv -> LIF1
- LIF0: top-k% firing strategy (no threshold)
- LIF1: standard threshold firing, outputs both spike and membrane potential

### Spike-Driven Attention (SDA)
- Replaces all multiplications with accumulations
- H_Att = g_temp_spike * g_ch_float * g_spa_spike
        + g_ch_spike * g_spa_float * g_temp_spike
        + g_spa_spike * g_temp_float * g_ch_spike + H
- Energy: 0 MUL, 5.82E6 AC = 5.24 uJ (vs 21.8 uJ for TCSA)

### Preference Coefficient Update (Algorithm 1)
- alpha_hat_t = (1/B) * sum(g_{t,b,temp}^float)  -- batch average
- alpha_t = 0.5 * (alpha_bar_t + alpha_hat_t)     -- temporal smoothing

## Backbone Detectors (TDE is a plug-in for these)
| Detector | Params | Source |
|----------|--------|--------|
| SpikeYOLO | 23.2M | ECCV 2024, BICLab/SpikeYOLO |
| EMS-YOLO | 33.9M | ICCV 2023, BICLab/EMS-YOLO |
| Meta-SpikeFormer | 16.7M | 2024, transformer-based SNN |

## Hyperparameters

### LIF Neuron
- Membrane decay factor beta: 0.25
- Firing threshold V_th: 1.0 (standard)
- Soft reset mechanism
- Timesteps T: 4

### Training (from SpikeYOLO baseline)
- Optimizer: SGD
- Learning rate: 0.01
- Batch size: 40 (4x V100; scale for L4)
- Epochs: 300 (COCO/VOC), 50 (neuromorphic)
- Data augmentation: Mosaic
- Framework: Ultralytics-based with spikingjelly

### TDE-Specific
- SE kernel size k: 3x3
- TDE parameter overhead: ~0.26M additional params
- Preference coefficient alpha: dynamically updated per timestep

## Datasets

### PASCAL VOC
- VOC2007: 9,963 images, 20 classes
- VOC2012: 11,530 images, 20 classes
- Static images repeated across T timesteps (direct encoding)
- YOLO-format labels

### EvDET200K (Neuromorphic)
- 10,054 event video streams, 202,260 annotations
- 10 object classes
- Event streams accumulated into frames R^(1 x H x W)
- Focus: small object detection, dynamic backgrounds

## Metrics
- mAP@50 (VOC-style)
- mAP@50:95 (COCO-style)
- Energy consumption (uJ at 45nm, 32-bit)
- Parameter count (M)

## Key Results
| Dataset | Model | Baseline | +TDE-SDA | Delta |
|---------|-------|----------|----------|-------|
| VOC2007 | SpikeYOLO | 51.7% | 56.2% | +4.5% |
| VOC2007 | EMS-YOLO | 59.8% | 61.3% | +1.5% |
| EvDET200K | SpikeYOLO | 46.5% | 47.2% | +0.7% |
| EvDET200K | EMS-YOLO | 44.9% | 45.9% | +1.0% |
| VOC combined | Best | -- | 57.7% mAP@50-95 | SOTA |
| EvDET200K | Best | -- | 47.6% mAP@50-95 | SOTA |

## Dependencies
- Python 3.11
- PyTorch 2.x (cu128)
- spikingjelly 0.0.0.0.14
- ultralytics (modified YOLO framework)
- torchvision
- numpy, scipy, opencv-python

## Model Requirements
- No pretrained weights required (direct training from scratch)
- SpikeYOLO / EMS-YOLO architecture definitions needed
- spikingjelly LIF neuron implementations
