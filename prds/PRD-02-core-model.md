# PRD-02: Core Model

> Status: TODO
> Module: 33_TDE
> Depends on: PRD-01

## Objective
Implement the full TDE architecture: LIF neurons, Spiking Encoder (SE),
Attention Gating Module (AGM) with both TCSA and SDA variants, and a
SpikeYOLO-style backbone detector.

## Components

### 1. LIF Neuron (src/anima_tde/neurons.py)
- Leaky Integrate-and-Fire with surrogate gradient
- Membrane decay beta = 0.25, threshold V_th = 1.0
- Soft reset mechanism
- LIF0 variant: top-k% firing (no threshold)
- LIF1 variant: standard threshold, dual output (spike + membrane)

### 2. Spiking Encoder (src/anima_tde/model.py)
- Input: image R^(C_in x H x W)
- Output: spike stream {0,1}^(T x C_out x H x W)
- Conv-BN block for t=0
- alpha-weighted mixing for t>0
- LIF conversion to binary spikes

### 3. Attention Gating Module (src/anima_tde/model.py)
- Temporal attention branch
- Channel attention branch
- Spatial attention branch
- TCSA variant (standard multiply)
- SDA variant (accumulation-only, energy-efficient)

### 4. SNN Backbone (src/anima_tde/model.py)
- SpikeYOLO-style architecture
- Spiking residual blocks
- Multi-scale feature pyramid
- Detection head (3 scales)

## Acceptance Criteria
- Forward pass works: input (B, 3, H, W) -> list of detection tensors
- Parameter count within 10% of paper (23.6M for SpikeYOLO+TDE-SDA)
- Both TCSA and SDA variants produce valid output
- Gradient flows through surrogate gradient
