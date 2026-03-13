# Desert Terrain Semantic Segmentation Using Synthetic Data
**Team:** AI LONE STARS

## 1. Introduction
Autonomous navigation in off-road environments requires robust, pixel-level terrain classification. This project addresses the challenge of segmenting 10 distinct desert terrain classes, including rare and small objects like Logs and Rocks. By training on high-fidelity synthetic data, we developed a model capable of generalizing to complex, unseen environments, providing a critical perception layer for autonomous vehicles navigating hazardous terrains.

## 2. Approach
We employed a multi-stage refinement strategy using the **DeepLabV3+** architecture. 

### 3-Phase Training Strategy:
1.  **Phase 1: Foundation (52.75% mIoU)**
    *   Backbone: ResNet34
    *   Loss: Cross-Entropy (CE)
    *   Focused on establishing a stable baseline and preventing class collapse.
2.  **Phase 2: Rare Class Boost (56.62% mIoU)**
    *   Integrated **WeightedRandomSampler** for 4x oversampling of rare classes (Logs, Rocks).
    *   Switched to **Combined CE + Dice Loss** to refine boundaries.
    *   Result: Significant jumps in Logs (+5.1%) and Rocks (+7.5%).
3.  **Phase 3: High-Resolution fine-tuning (59.11% mIoU)**
    *   Increased input resolution from 384x384 to **512x512**.
    *   Used **CosineAnnealingWarmRestarts** for fine-grained convergence.
    *   Implemented 4-Augmentation TTA for local validation boost.
4.  **Final Step: Cloud ResNet101 & Ultra TTA (64.50% mIoU)**
    *   Transferred training to **Google Colab (Tesla T4)** to utilize a deeper **ResNet101** backbone.
    *   Increased resolution to **640x640** for superior small-object recall.
    *   Implemented an **8-Augmentation Ultra TTA** (Scales + Flips) for final inference.

## 3. Challenges & Solutions
*   **Challenge 1: Model Collapse**
    *   Initially, the model predicted only the dominant "Sky" class (mIoU=2.37%).
    *   **Solution:** Removed Dice loss for the first 50 epochs, increased learning rate to 1e-3, and verified mask remapping indices.
*   **Challenge 2: Extreme Class Imbalance**
    *   Sky covers ~37% of pixels while Logs cover only ~0.3%.
    *   **Solution:** Implemented inverse-frequency class weights capped at 8.0x and targeted oversampling of minority classes via a custom sampler.
*   **Challenge 3: Limited GPU Resources**
    *   Training was limited to 6GB VRAM on a laptop RTX 4050.
    *   **Solution:** Used Mixed Precision (AMP), Gradient Accumulation (effective batch size 8), and chose a ResNet34 backbone for optimal speed-accuracy trade-off.

The model showed consistent growth across all metrics. Notably, the **Logs** class improved from **21.90% to 53.91% IoU** through our multi-phase optimization, contributing to a final mIoU of **64.50%**.

### Summary Metrics:
| Phase | mIoU | Best Class | Worst Class |
| :--- | :--- | :--- | :--- |
| Phase 1 | 52.75% | Sky (97.07%) | Logs (21.90%) |
| Phase 2 | 56.62% | Sky (97.57%) | Logs (27.00%) |
| Phase 3 | 59.11% | Sky (97.86%) | Logs (31.45%) |
| **Final (Apex)**| **64.50%**| **Sky (97.67%)** | **Ground Clutter** |

### Per-Class Progression:
| Class | Phase 1 | Phase 2 | Phase 3 | Final (Ultra TTA) |
| :--- | :--- | :--- | :--- | :--- |
| Trees | 77.13% | 80.11% | 82.35% | **85.30%** |
| Lush Bushes | 64.66% | 66.34% | 67.66% | **69.26%** |
| Dry Grass | 63.79% | 65.89% | 66.97% | **69.32%** |
| Dry Bushes | 35.40% | 42.57% | 42.93% | **48.83%** |
| Ground Clutter | 31.56% | 33.23% | 35.43% | **38.97%** |
| Flowers | 53.75% | 58.41% | 60.56% | **65.76%** |
| Logs | 21.90% | 27.00% | 31.45% | **53.91%** |
| Rocks | 29.85% | 37.33% | 39.37% | **47.89%** |
| Landscape | 52.36% | 57.77% | 59.58% | **68.08%** |
| Sky | 97.07% | 97.57% | 97.86% | **97.67%** |

## 5. Key Insights
1.  **Oversampling is Critical:** Targeted oversampling was the single biggest contributor to mIoU gains (+3.87% in Phase 2).
2.  **Resolution Matters:** Moving to 512x512 resolution in Phase 3 allowed the model to recover small object details (Logs +2.3%).
3.  **Loss Scheduling:** Dice loss is highly sensitive; adding it only after the model's CE baseline stabilized prevented divergence.

## 6. Real-World Impact
This project demonstrates that high-fidelity synthetic data is a viable and cost-effective alternative to real-world data collection. The multi-phase refinement approach ensures that even rare objects—critical for obstacle avoidance in autonomous driving—are accurately detected without sacrificing performance on common classes.

## 7. Performance Visualization

### Progression & Metrics
![Per-Class IoU Progression](laptop/runs/final/iou_chart.png)
*Figure 1: Comparison of IoU across different training phases.*

### Confusion Analysis
![Confusion Matrix](laptop/runs/final/confusion_matrix.png)
*Figure 2: Normalized Confusion Matrix showing minor misclassifications between Bushes and Dry Grass.*

### Qualitative Results
#### Best Validation Examples:
![Best Example 1](laptop/runs/final/best_examples/rank01_Color_000305.png)
![Best Example 2](laptop/runs/final/best_examples/rank02_Color_000300.png)

#### Challenge Cases (Small Logs/Rocks):
![Worst Case](laptop/runs/final/worst_examples/worst01_Color_000306.png)
