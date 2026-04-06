# PRD-06: Export Pipeline

> Status: TODO
> Module: 33_TDE
> Depends on: PRD-05

## Objective
Export trained models to production formats: safetensors, ONNX, TensorRT.

## Components

### 1. Safetensors Export
- Convert best.pth to safetensors format
- Verify weight fidelity (max diff < 1e-6)

### 2. ONNX Export
- Export with opset 17
- Dynamic batch size
- Fixed T=4 timesteps (unrolled)
- Input: (B, 3, H, W), Output: list of detection tensors
- Validate with onnxruntime

### 3. TensorRT Export
- Use shared TRT toolkit: /mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py
- FP16 and FP32 variants
- Benchmark latency

### 4. HuggingFace Push
- Repo: ilessio-aiflowlab/project_tde-checkpoint
- Include: safetensors, ONNX, config, metrics

## Special Considerations
- SNN with LIF neurons must be unrolled across T timesteps for ONNX
- Spike operations (binary threshold) need custom ONNX ops or approximation
- TensorRT may need plugin for LIF neuron

## Acceptance Criteria
- Safetensors loads and produces identical output to .pth
- ONNX validates with onnx.checker
- ONNX runtime inference matches PyTorch within tolerance
- TRT FP16 and FP32 exported successfully
