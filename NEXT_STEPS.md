# NEXT_STEPS.md
> Last updated: 2026-04-06
> MVP Readiness: 75%

## Done
- [x] Paper analysis complete (arXiv 2512.02447)
- [x] CLAUDE.md with full architecture + hyperparameters
- [x] ASSETS.md with dataset/model inventory
- [x] PRD.md with 7-PRD build plan
- [x] 7 PRD files in prds/
- [x] tasks/INDEX.md with granular tasks
- [x] pyproject.toml with hatchling + cu128
- [x] configs/paper.toml + debug.toml + evdet200k.toml
- [x] anima_module.yaml
- [x] src/anima_tde/ -- full Python package
- [x] LIF neurons (standard, LIF0 top-k%, LIF1 dual-output)
- [x] Spiking Encoder (SE) with learnable alpha
- [x] Attention Gating Module (AGM) -- temporal, channel, spatial
- [x] Spike-Driven Attention (SDA) -- accumulation-only, 0 MUL
- [x] SpikeYOLO backbone with multi-scale detection head
- [x] Detection loss (CIoU + BCE obj + BCE cls)
- [x] Training pipeline with cosine warmup + early stopping
- [x] VOC + EvDET200K dataset loaders
- [x] Evaluation pipeline (mAP@50, mAP@50:95, energy)
- [x] Serving node (FastAPI) + Docker
- [x] **Custom CUDA kernels**: fused_lif_forward/backward, fused_sda, fused_alpha_mix
- [x] **Shared CUDA integration**: detection_ops (NMS, IoU), fused_image_preprocess
- [x] Detection decoder with CUDA-accelerated NMS
- [x] Export pipeline: pth -> safetensors -> ONNX -> TRT FP16 -> TRT FP32
- [x] ONNX-compatible pooling (replaced AdaptiveMaxPool2d)
- [x] .venv with torch cu128, spikingjelly, all deps
- [x] VOC2007 extracted + YOLO format (4510 train, 501 val, 4952 test)
- [x] 33/33 tests passing, lint clean
- [x] GPU smoke test passed (17.13M params, forward+backward OK)

## In Progress
- [ ] Waiting for user to confirm GPU availability for training

## TODO
- [ ] Run full training on VOC2007 (300 epochs, nohup+disown)
- [ ] Evaluate on VOC2007 test split (mAP@50, mAP@50:95)
- [ ] Generate TRAINING_REPORT.md
- [ ] Export all 5 formats from best checkpoint
- [ ] Push to HuggingFace (ilessio-aiflowlab/project_tde-checkpoint)
- [ ] Add VOC2012 data when available
- [ ] Add EvDET200K data when available
- [ ] Docker build + health check

## Blocking
- Need GPU assignment from user before training
- VOC2012 downloading (user will notify)
- EvDET200K downloading from Baidu (slow, user will notify)

## Downloads Needed
- PASCAL VOC 2012: ~2GB -- user downloading
- EvDET200K: ~10GB -- user downloading from Baidu
