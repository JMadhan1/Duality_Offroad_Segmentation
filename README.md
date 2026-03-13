# 🏜️ Duality AI: Offroad Semantic Segmentation Challenge

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Summary
This repository contains our submission for the **Duality AI Offroad Semantic Segmentation Challenge**. The goal is to perform pixel-wise terrain classification for autonomous off-road navigation in desert environments.

- **Objective:** Segmentation of 10 distinct terrain classes.
- **Dataset:** High-fidelity synthetic imagery provided by the Duality AI simulator.
- **Key Challenge:** Bridging the gap between synthetic training data and diverse, unseen test environments.

---

## 🚀 Our Approach: Multi-Phase Training Strategy

We adopted a progressive training pipeline that scaled from a local development environment to high-performance cloud GPUs.

### 💻 Phase 1: Base Training (Local Laptop)
- **Hardware:** RTX 4050 (6GB VRAM)
- **Model:** DeepLabV3+ with **ResNet-50** backbone
- **Resolution:** 384x384, Batch Size: 4
- **Strategy:** CrossEntropyLoss with class weights to handle initial imbalance.
- **Result:** **52.75% mIoU**

### 📈 Phase 2: Minority Class Boost
- **Focus:** Tackling the "Logs" and "Rocks" classes which had near-zero IoU.
- **Loss:** Hybrid **CE + Dice Loss** (0.5/0.5 ratio).
- **Sampling:** `WeightedRandomSampler` (Logs 4x, Rocks 3x oversampling).
- **Augmentation:** Introduced **CoarseDropout** to force the model to learn from partial context.
- **Result:** **56.62% mIoU**

### 🖼️ Phase 3: Resolution & Scheduler Push
- **Resolution:** Increased to **512x512**.
- **Optimization:** Gradient accumulation (Effective batch=8) + **CosineAnnealingWarmRestarts**.
- **Result:** **58.02% mIoU** (59.11% with TTA).

### ⚡ Phase 4: The Final Cloud Push (Google Colab)
- **Hardware:** Tesla T4 (15GB VRAM)
- **Model:** Upgraded to **ResNet-101** backbone.
- **Resolution:** **640x640**, Batch Size: 8.
- **Ensemble:** 5-Augmentation TTA (Original, H-Flip, V-Flip, 0.75x Scale, 1.25x Scale).
- **Final Validation mIoU:** **63.89%**

---

## 📊 Results & Performance

### Final Per-Class IoU (Validation):
| Class | IoU |
| :--- | :--- |
| 🌳 Trees | 85.00% |
| 🌿 Lush Bushes | 69.17% |
| 🌾 Dry Grass | 69.10% |
| 🍂 Dry Bushes | 48.35% |
| 🪨 Ground Clutter | 38.47% |
| 🌸 Flowers | 65.48% |
| 🪵 Logs | 50.85% |
| 💎 Rocks | 47.28% |
| 🏔️ Landscape | 67.60% |
| ☁️ Sky | 97.64% |
| **🏆 Mean IoU** | **63.89%** |

### Progression Journey:
| Stage | mIoU | Key Change |
| :--- | :--- | :--- |
| Phase 1 | 52.75% | Base DeepLabV3+ setup |
| Phase 2 | 56.62% | Class-weighted oversampling + Dice Loss |
| Phase 3 | 59.11% | 512px + 4-Aug TTA |
| **Phase 4** | **63.89%** | **ResNet-101 + 640px + 5-Aug TTA** |

---

## 🛠️ Key Challenges Solved
1. **Model Collapse:** Initially, the model overfit to the "Landscape" background. This was solved by applying dynamic class weights and integrating Dice Loss to ensure gradient flow for smaller objects.
2. **The "Log" Problem:** Minority classes like Logs went from **3.34% ➡️ 50.85% IoU** through aggressive oversampling and custom augmentation.
3. **VRAM Optimization:** We balanced deep architectures with memory constraints using mixed-precision training (`torch.amp`) and gradient accumulation.
4. **Generalization:** Implemented a heavy Albumentations pipeline (Blur, Noise, Distortion) to ensure the model learned robust features rather than synthetic artifacts.

---

## 🏗️ Repository Structure
```text
.
├── README.md               # You are here
├── hackathon_report.md      # Detailed technical analysis
├── requirements.txt         # Environment dependencies
├── laptop/                  # Local development scripts
│   ├── train.py             # Phase 1 & 2 logic
│   ├── phase3_train.py      # High-res training
│   └── test_tta.py          # Local inference
├── colab/                   # Cloud resources
│   └── hackathon_training.ipynb
├── predictions/             # Final raw uint16 masks
└── visualizations/          # Performance charts & Heatmaps
```

---

## 💻 Tech Stack
- **Framework:** PyTorch
- **Architecture:** DeepLabV3+ (segmentation-models-pytorch)
- **Augmentation:** Albumentations
- **Techniques:** Mixed Precision, TTA (ttach), Weighted Sampling

## 🏃 How to Run
1. **Install dependencies:** `pip install -r requirements.txt`
2. **Local Inference:** `python laptop/test_tta.py`
3. **Cloud Training:** Upload `colab/hackathon_training.ipynb` to Google Colab.

---
**Team:** [YOUR TEAM NAME]
*Developed for the Duality AI 2026 Hackathon Challenges*
