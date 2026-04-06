# ASSETS.md -- TDE Asset Inventory

> Module: 33_TDE (Wave 9 Defense)
> Last updated: 2026-04-05

## Datasets Required

### 1. PASCAL VOC 2007
- **Status**: NOT on disk -- needs download
- **Size**: ~900MB
- **Path (target)**: /mnt/forge-data/datasets/voc/VOC2007/
- **Source**: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/
- **Format**: JPEG images + XML annotations (convert to YOLO format)
- **Classes**: 20 (aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair,
  cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa,
  train, tvmonitor)
- **Splits**: trainval (5,011) + test (4,952)

### 2. PASCAL VOC 2012
- **Status**: NOT on disk -- needs download
- **Size**: ~2GB
- **Path (target)**: /mnt/forge-data/datasets/voc/VOC2012/
- **Source**: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
- **Format**: JPEG images + XML annotations (convert to YOLO format)
- **Classes**: 20 (same as VOC2007)
- **Splits**: trainval (11,540)

### 3. EvDET200K (Neuromorphic Event Dataset)
- **Status**: NOT on disk -- needs download
- **Size**: ~10GB (estimated)
- **Path (target)**: /mnt/forge-data/datasets/evdet200k/
- **Source**: Check paper references / authors' release
- **Format**: Event streams accumulated to frames + YOLO-format labels
- **Classes**: 10 object categories
- **Samples**: 10,054 streams, 202,260 annotations

## Pretrained Models Required

### 4. spikingjelly Library
- **Status**: NOT installed -- needs pip install
- **Version**: 0.0.0.0.14
- **Install**: `uv pip install spikingjelly==0.0.0.0.14`
- **Purpose**: LIF neuron implementation, surrogate gradient functions

### 5. SpikeYOLO Backbone Weights (optional, for comparison)
- **Status**: NOT on disk
- **Source**: https://github.com/BICLab/SpikeYOLO
- **Purpose**: Baseline comparison (can train from scratch)

### 6. EMS-YOLO Backbone Weights (optional, for comparison)
- **Status**: NOT on disk
- **Source**: https://github.com/BICLab/EMS-YOLO
- **Purpose**: Baseline comparison (can train from scratch)

## Shared Infrastructure Assets (already on disk)

### CUDA Extensions
- Fused image preprocess: /mnt/forge-data/shared_infra/cuda_extensions/fused_image_preprocess/
- Detection ops: /mnt/forge-data/shared_infra/cuda_extensions/detection_ops/
- Install: `uv pip install /mnt/forge-data/shared_infra/cuda_extensions/wheels_py311_cu128/*.whl`

### Models on Disk (not directly needed but available)
- YOLOv5l6: /mnt/forge-data/models/yolov5l6.pt (reference only)
- YOLO11n: /mnt/forge-data/models/yolo11n.pt (reference only)

## Download Commands

```bash
# VOC2007
cd /mnt/forge-data/datasets/
mkdir -p voc && cd voc
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar

# VOC2012
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_11-May-2012.tar

# spikingjelly
cd /mnt/forge-data/modules/05_wave9/33_TDE
source .venv/bin/activate
uv pip install spikingjelly==0.0.0.0.14

# EvDET200K -- check official release for download link
# https://github.com/Mortal825/TDE or paper supplementary
```

## Storage Estimate
| Asset | Size | Disk |
|-------|------|------|
| VOC2007 | ~900MB | /mnt/forge-data/datasets/ |
| VOC2012 | ~2GB | /mnt/forge-data/datasets/ |
| EvDET200K | ~10GB | /mnt/forge-data/datasets/ |
| spikingjelly | ~50MB | .venv/ |
| Checkpoints | ~500MB | /mnt/artifacts-datai/ |
| **Total** | **~13.5GB** | |
