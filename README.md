# Desert Terrain Semantic Segmentation

This repository contains the full pipeline for the Desert Terrain Semantic Segmentation hackathon. The goal is to perform pixel-level classification of 10 desert mountain terrain classes using synthetic training data.

## Results
| Metric | Score |
| :--- | :--- |
| **Final mIoU (TTA)** | **59.11%** |
| Best Class | Sky (97.96%) |
| Most Improved | Rocks (+11.48%) |

## Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Inference with TTA:**
   ```bash
   python laptop/test_tta.py
   ```
   This will load the best checkpoint from `laptop/runs/phase3/best_model.pth` and generate final predictions in `laptop/runs/final/`.

## Project Structure
*   `laptop/`: Root for all local training code and logs.
    *   `train.py`: Phase 1 training (Base CE loss).
    *   `phase2_train.py`: Phase 2 fine-tuning (Dice loss + Oversampling).
    *   `phase3_train.py`: Phase 3 high-resolution (512x512) fine-tuning.
    *   `test_tta.py`: Final evaluation with Test-Time Augmentation (TTA).
    *   `src/`: Core logic for datasets, models, metrics, and augmentations.
    *   `runs/`:
        *   `phase2/`, `phase3/`: Intermediate checkpoints and logs.
        *   `final/`: Final predictions, overlays, and report assets.
*   `colab/`: Google Colab training notebook.
*   `predictions/`: Ready-to-submit test set predictions.
*   `hackathon_report.md`: Detailed methodology and results analysis.

## Hardware
*   **Training:** NVIDIA RTX 4050 Laptop GPU (6GB VRAM), Intel i7-13620H, 16GB RAM.
*   **Alternative:** Fully compatible with Google Colab (T4/A100) via `hackathon_training.ipynb`.

## Training Strategy
We used a 3-phase curriculum learning strategy:
1. **Base Training:** Cross-entropy loss on 384x384.
2. **Boundary Refinement:** Combined CE + Dice loss with rare class oversampling.
3. **Resolution Push:** Fine-tuning on 512x512 with Cosine LR scheduling.
4. **Inference Boost:** TTA with scale and flip variants.

---
**Report:** See [hackathon_report.md](hackathon_report.md) for full technical details.
