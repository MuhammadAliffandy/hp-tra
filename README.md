# 🎨 Neural Gesture Mannequin — Multi-Task Volumetric PoseNet

**Author:** Muhammad Aliffandy (Apple Developer Academy & PukulEnam AI/ML Engineer)  
**Target Architecture:** Multi-Task Direct Coordinate & Volumetric Regression $\to$ Parametric 3D Mesh Recovery (SMPL)  
**Hardware Platforms:** NVIDIA DGX Cluster (H100) for Training $\to$ Apple Neural Engine (ANE) via CoreML Float16 for iPadOS Deployment.

---

## 🏆 Research Evolution: Challenge Based Learning (CBL) Iterations

Proyek ini mendokumentasikan perjalanan rekayasa AI bertahap (*learning journey*) yang memecahkan limitasi Apple Vision Framework bawaan secara berkesinambungan:

| Milestone Riset | Fokus Rekayasa & Arsitektur | Dataset & Komputasi | Hasil & Evaluasi Kritis | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Iterasi 1 (v1.0)** | **2D Skeleton Rig & Heuristic Volumetric Mannequin**<br>Direct Coordinate Regression + Masked Smooth L1 Loss | **MS-COCO 2017 (114.554 Foto)**<br>NVIDIA DGX H100 (80 Epochs, 3.2 Jam) | **PCK@0.1: 84.7% \| Pixel Error: 8.8 px**<br>*Temuan Riset:* Mampu mendeteksi sendi dinamis secara presisi, namun representasi volumetrik masih mengandalkan formula silinder 2D OpenCV. | ✅ **Archived (`v1.0-iterasi1-2d-skeleton`)** |
| **Iterasi 2 (v2.0)** | **True Parametric 3D Human Mesh Recovery (HMR)**<br>Regresi langsung parameter bentuk tubuh ($\beta \in \mathbb{R}^{10}$) dan rotasi sendi 3D ($\theta \in \mathbb{R}^{72}$) SMPL | **COCO-EFT (114.554 Foto)**<br>Fine-Tuning di atas Backbone MobileNetV3 Iterasi 1 | **Rekonstruksi Volume 3D Sejati (End-to-End)**<br>Memprediksi ketebalan massa otot dan perspektif 3D murni dari representasi laten AI. | 🚧 **Active Research Sprint** |

---

## 🌟 Dual-Mode Architectural Overview (Iterasi 1 Baseline)

Unlike standard pose estimators that only produce flat 2D stick figures, **Neural Gesture Mannequin** is designed specifically for digital artists and webtoon creators, providing two native viewing modes:

1. **Mode 1: Skeleton Wireframe (Fig. 2 Kiri)**  
   17 continuous joint coordinates predicting the *Line of Action*, dynamic gesture rhythm, and physical balance.
2. **Mode 2: Volumetric 3D Mannequin (Fig. 2 Kanan)**  
   3D tapered cylinders, cross-contour ellipses (*cincin kontur perspektif*), and torso/pelvis orientation boxes capturing anatomical muscle volumes and foreshortening.

```text
                       [Input Citra Pose 256x256]
                                    │
               [Pretrained MobileNetV3 Backbone (Iterasi 1)]
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        [Head 1: Koordinat Sendi]           [Head 2: Radius Volume]
       17 Titik Sendi (x, y) 34 output      6 Parameter Silinder Otot (r)
                  │                                   │
                  ▼                                   ▼
          MODE 1: SKELETON                    MODE 2: VOLUME MANNEQUIN
      (Rangka Garis Aksi & Sendi)         (Silinder 3D, Cincin Elips & Kotak)
                  │
                  ▼
          [Step 5: Anime Lineart / Atari Pose Base Generator]
```

---

## 🚀 Quick Start on NVIDIA DGX Cluster (PukulEnam)

### 1. Requirements & Environment (Python 3.10.2)
```bash
# Sesuai standar DGX kantor PukulEnam:
pip install -r requirements.txt
```

### 2. Auto-Download Complete MS-COCO Keypoints Dataset (1-Command)
```bash
# Otomatis mengunduh seluruh dataset lengkap (train2017 + val2017 + annotations ~20 GB):
python3 download_coco.py
```

### 3. One-Command Background Execution on DGX (100% Disconnect-Proof)
```bash
# Menjalankan di GPU 1 secara default di background (nohup + disown):
./run_dgx.sh
```

### 4. Monitoring & Process Management
```bash
# Monitor log pelatihan secara real-time:
tail -f train_dgx.log

# Snapshot visual per-epoch tersimpan otomatis di:
# ./checkpoints_dgx/snapshots/snapshot_epoch_XXX.png
```

---

## 💻 Testing & Inference on Google Colab
Notebook inferensi interaktif untuk mencoba foto kustom tersedia di:  
👉 **`test_inference_colab.ipynb`** (Mendukung visualisasi 4-panel dan ekstraksi garis manekin anime solo ~150 MB).

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
