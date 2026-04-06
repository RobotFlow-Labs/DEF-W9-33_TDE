# NEXT_STEPS.md
> Last updated: 2026-04-05
> MVP Readiness: 25%

## Done
- [x] Paper analysis complete (arXiv 2512.02447)
- [x] CLAUDE.md with full architecture + hyperparameters
- [x] ASSETS.md with dataset/model inventory
- [x] PRD.md with 7-PRD build plan
- [x] 7 PRD files in prds/
- [x] tasks/INDEX.md with granular tasks
- [x] pyproject.toml with hatchling + cu128
- [x] configs/paper.toml + debug.toml
- [x] anima_module.yaml
- [x] src/anima_tde/ -- full Python package (model, dataset, train, evaluate, losses, utils, neurons)
- [x] scripts/train.py + scripts/evaluate.py
- [x] tests/test_model.py + tests/test_dataset.py
- [x] Dockerfile.serve + docker-compose.serve.yml

## In Progress
- [ ] Nothing currently in progress

## TODO
- [ ] Create .venv and install dependencies (uv sync)
- [ ] Download PASCAL VOC 2007/2012 datasets
- [ ] Download EvDET200K dataset
- [ ] Run ruff check and fix any lint issues
- [ ] Run pytest smoke tests
- [ ] PRD-04: Full training pipeline validation
- [ ] PRD-05: Evaluation on test set
- [ ] PRD-06: ONNX + TRT export
- [ ] PRD-07: Docker build + serve test

## Blocking
- VOC and EvDET200K datasets not yet on disk
- spikingjelly not yet installed in venv

## Downloads Needed
- PASCAL VOC 2007: ~900MB -- see ASSETS.md
- PASCAL VOC 2012: ~2GB -- see ASSETS.md
- EvDET200K: ~10GB -- check paper authors' release
- spikingjelly: `uv pip install spikingjelly==0.0.0.0.14`
