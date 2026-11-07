# 🎨 Neural Gesture Mannequin — Multi-Task Volumetric PoseNet

**Author:** Muhammad Aliffandy (Apple Developer Academy & PukulEnam AI/ML Engineer)  
**Target Architecture:** Multi-Task Direct Coordinate & Volumetric Regression (Pretrained MobileNetV3 / ResNet Backbone)  
**Hardware Platforms:** NVIDIA DGX Cluster (A100/H100) for Training $\to$ Apple Neural Engine (ANE) via CoreML Float16 for iPadOS Deployment.

---

## 🌟 Dual-Mode Architectural Overview

Unlike standard pose estimators that only produce flat 2D stick figures, **Neural Gesture Mannequin** is designed specifically for digital artists and webtoon creators, providing two native viewing modes:

1. **Mode 1: Skeleton Wireframe (Fig. 2 Kiri)**  
   17 continuous joint coordinates predicting the *Line of Action*, dynamic gesture rhythm, and physical balance.
2. **Mode 2: Volumetric 3D Mannequin (Fig. 2 Kanan)**  
   3D tapered cylinders, cross-contour ellipses (*cincin kontur perspektif*), and torso/pelvis orientation boxes capturing anatomical muscle volumes and foreshortening.

```text
                       [Input Citra Pose 256x256]
                                    │
                     [Pretrained Vision Backbone]
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        [Head 1: Koordinat Sendi]           [Head 2: Radius Volume]
       17 Titik Sendi (x, y) 34 output      6 Parameter Silinder Otot (r)
                  │                                   │
                  ▼                                   ▼
          MODE 1: SKELETON                    MODE 2: VOLUME MANNEQUIN
      (Rangka Garis Aksi & Sendi)         (Silinder 3D, Cincin Elips & Kotak)
```

---

## 🚀 Quick Start on NVIDIA DGX Cluster (PukulEnam)

### 1. Requirements & Environment (Python 3.10.2)
```bash
# Sesuai standar DGX kantor PukulEnam (numpy < 2 ABI protection):
pip install -r requirements.txt
```

### 2. Auto-Download MS-COCO Keypoints Dataset
```bash
# Opsi A: Download Val Split (~1.2 GB, ideal untuk smoke test cepat):
python3 download_coco.py --split val

# Opsi B: Download Full Training Set (~20 GB untuk main training):
python3 download_coco.py --split all --clean-zip
```

### 3. One-Command Background Execution on DGX
```bash
# Menjalankan di GPU 0 secara otomatis di background (nohup):
./run_dgx.sh

# Atau jika ingin memilih nomor GPU tertentu (misal GPU 5):
./run_dgx.sh 5
```

### 4. Monitoring & Process Management
```bash
# Monitor log pelatihan secara real-time:
tail -f train_dgx.log

# Snapshot visual per-epoch tersimpan otomatis di:
# ./checkpoints_dgx/snapshots/snapshot_epoch_XXX.png
```

---

## 💻 Local Testing on macOS (MPS / CPU)

Untuk pengujian cepat di laptop Mac tanpa server DGX:
```bash
./run_local.sh
```

---

## 🍏 One-Click Apple CoreML Export for iPadOS
Setelah file bobot terbaik `volumetric_posenet_best.pth` dihasilkan dari server DGX:
```bash
python3 src/export_coreml.py \
    --checkpoint ./checkpoints_dgx/volumetric_posenet_best.pth \
    --output VolumetricPoseNet_ANE.mlpackage \
    --backbone mobilenet_v3_large
```
File `.mlpackage` Float16 siap dimasukkan langsung ke project Xcode (Sketsain) untuk berjalan di Apple Neural Engine (ANE) iPad Pro / Air dengan latensi $<2\text{ ms}$.
