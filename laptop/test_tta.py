"""
test_tta.py — Final Evaluation & Submission with Test-Time Augmentation (TTA).

Performs:
1. Validation evaluation (with vs without TTA)
2. Test set inference with TTA
3. Report generation:
   - Confusion matrix
   - Per-class IoU bar chart (Comparison)
   - Best/Worst visualization examples
   - Detailed results.txt summary

Strategy:
- Original + H-Flip + 0.75x + 1.25x (Average of softmax probabilities)
- Mixed precision inference (AMP)
- UTF-8 encoding for all file exports
"""

import os
import sys
import time
import json
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Local src imports
sys.path.insert(0, str(Path(__file__).parent))
from src.dataset import OffRoadDataset, CLASS_NAMES, IGNORE_INDEX, INV_CLASS_MAP
from src.model import build_deeplabv3plus
from src.metrics import SegmentationMetrics
from src.utils import set_seed, mask_to_rgb, make_legend_patches

# --------------------------------------------------------------------------- #
# Multi-Phase Baseline Data
# --------------------------------------------------------------------------- #
PHASE1_IOU = [0.7713, 0.6466, 0.6379, 0.3540, 0.3156, 0.5375, 0.2190, 0.2985, 0.5236, 0.9707]
PHASE2_IOU = [0.8011, 0.6634, 0.6589, 0.4257, 0.3323, 0.5841, 0.2700, 0.3733, 0.5777, 0.9757]
# Best Phase 3 results from your training log
PHASE3_IOU = [0.8235, 0.6766, 0.6697, 0.4293, 0.3543, 0.6056, 0.2934, 0.3937, 0.5958, 0.9786]

# --------------------------------------------------------------------------- #
# TTA Module
# --------------------------------------------------------------------------- #
class TTAInference:
    """Handles TTA logic: Original, Flip, 0.75x, 1.25x."""
    def __init__(self, model, device, use_amp=True):
        self.model = model
        self.device = device
        self.use_amp = use_amp

    def predict(self, image_tensor):
        """
        Args:
            image_tensor: [1, 3, H, W] normalized tensor
        Returns:
            softmax_avg: [1, C, H, W]
        """
        B, C, H, W = image_tensor.shape
        accumulated_probs = []

        self.model.eval()
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
                # 1. Original
                logits = self.model(image_tensor)
                accumulated_probs.append(F.softmax(logits, dim=1))

                # 2. Horizontal Flip
                flipped_img = image_tensor.flip(-1)
                logits_flip = self.model(flipped_img)
                probs_flip = F.softmax(logits_flip, dim=1).flip(-1)
                accumulated_probs.append(probs_flip)

                # 3. 0.75x Scale
                sh, sw = int(H * 0.75), int(W * 0.75)
                # Ensure div by 32 for DeepLabV3+
                sh, sw = (sh // 32) * 32, (sw // 32) * 32
                sw = max(sw, 32); sh = max(sh, 32)
                
                img_075 = F.interpolate(image_tensor, size=(sh, sw), mode='bilinear', align_corners=False)
                logits_075 = self.model(img_075)
                probs_075 = F.softmax(logits_075, dim=1)
                probs_075 = F.interpolate(probs_075, size=(H, W), mode='bilinear', align_corners=False)
                accumulated_probs.append(probs_075)

                # 4. 1.25x Scale
                sh, sw = int(H * 1.25), int(W * 1.25)
                sh, sw = (sh // 32) * 32, (sw // 32) * 32
                
                img_125 = F.interpolate(image_tensor, size=(sh, sw), mode='bilinear', align_corners=False)
                logits_125 = self.model(img_125)
                probs_125 = F.softmax(logits_125, dim=1)
                probs_125 = F.interpolate(probs_125, size=(H, W), mode='bilinear', align_corners=False)
                accumulated_probs.append(probs_125)

        # Average of all probabilities
        softmax_avg = torch.stack(accumulated_probs).mean(dim=0)
        return softmax_avg

# --------------------------------------------------------------------------- #
# Visualisation Helpers
# --------------------------------------------------------------------------- #
def save_comparison_grid(image, gt, pred, save_path, title):
    """Saves RGB | GT | Pred side-by-side."""
    gt_rgb = mask_to_rgb(gt)
    pred_rgb = mask_to_rgb(pred)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image); axes[0].set_title("RGB Image"); axes[0].axis('off')
    axes[1].imshow(gt_rgb); axes[1].set_title("Ground Truth"); axes[1].axis('off')
    axes[2].imshow(pred_rgb); axes[2].set_title(f"TTA Prediction\n{title}"); axes[2].axis('off')
    
    patches = make_legend_patches(CLASS_NAMES)
    fig.legend(handles=patches, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.02), fontsize=9)
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(save_path, dpi=100)
    plt.close()

def plot_per_class_comparison(p1, p2, p3, tta, save_path):
    """Bar chart comparing all phases."""
    df = pd.DataFrame({
        'Class': CLASS_NAMES * 4,
        'IoU (%)': ([x*100 for x in p1] + [x*100 for x in p2] + 
                    [x*100 for x in p3] + [x*100 for x in tta]),
        'Phase': (['Phase 1'] * 10 + ['Phase 2'] * 10 + 
                  ['Phase 3'] * 10 + ['TTA Boost'] * 10)
    })
    
    plt.figure(figsize=(15, 7))
    sns.barplot(data=df, x='Class', y='IoU (%)', hue='Phase', palette='viridis')
    plt.axhline(y=np.mean(tta)*100, color='red', linestyle='--', alpha=0.6, label='Final mIoU')
    plt.title("Per-Class IoU Progression: Phase 1 → Phase 2 → Phase 3 → TTA", fontsize=14)
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=30)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

# --------------------------------------------------------------------------- #
# Main Logic
# --------------------------------------------------------------------------- #
def main():
    set_seed(42)
    
    # Paths
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    val_images = cfg['data']['val_images']
    val_masks  = cfg['data']['val_masks']
    test_images = cfg['data']['test_images']
    
    ckpt_path = 'runs/phase3/best_model.pth'
    final_dir = Path('runs/final')
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / 'predictions').mkdir(exist_ok=True)
    (final_dir / 'overlays').mkdir(exist_ok=True)
    (final_dir / 'best_examples').mkdir(exist_ok=True)
    (final_dir / 'worst_examples').mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using {device}")

    # Load Model
    print(f"[Model] Loading DeepLabV3+ ResNet34 from {ckpt_path}...")
    model = build_deeplabv3plus(backbone='resnet34', encoder_weights=None, num_classes=10).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()

    tta_engine = TTAInference(model, device, use_amp=True)

    # 1. Validation Evaluation
    val_ds = OffRoadDataset(val_images, val_masks, transform=None) # Raw images for manual control
    # But for standard metrics we need a transformed loader too (for "without TTA")
    from src.transforms import get_val_transforms
    val_loader = DataLoader(
        OffRoadDataset(val_images, val_masks, transform=get_val_transforms((512,512))),
        batch_size=1, shuffle=False
    )

    metrics_no_tta = SegmentationMetrics(10)
    metrics_tta    = SegmentationMetrics(10)
    
    val_details = [] # Store (iou, image, gt, pred, filename) for best/worst
    
    print("\n[Eval] Running Validation with TTA vs Base...")
    # Faster to use 1 loop for both
    # We use the val_transforms image for "No TTA" and raw image for TTA implementation
    val_norm_tfm = get_val_transforms((512, 512))
    
    inference_times = []

    for i in tqdm(range(len(val_ds)), desc="Validation Samples"):
        img_raw, gt = val_ds[i]
        filename = val_ds.image_files[i].name
        
        # Prep tensor for model
        input_tensor = val_norm_tfm(image=img_raw)['image'].unsqueeze(0).to(device)
        
        start_time = time.time()
        
        # A. Predict No-TTA
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                logits_base = model(input_tensor)
        
        # Upsample predictions to original GT size for accurate evaluation
        h_orig, w_orig = gt.shape
        logits_base_up = F.interpolate(logits_base, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
        pred_no_tta = logits_base_up.argmax(1).cpu().numpy()[0]
        
        # B. Predict TTA
        probs_tta = tta_engine.predict(input_tensor)
        # Upsample TTA probs back to GT size
        probs_tta_up = F.interpolate(probs_tta, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
        pred_tta  = probs_tta_up.argmax(1).cpu().numpy()[0]
        
        elapsed = time.time() - start_time
        inference_times.append(elapsed)

        # Update Metrics
        metrics_no_tta.update(pred_no_tta, gt)
        metrics_tta.update(pred_tta, gt)
        
        # Calculate sample IoU (mIoU for this specific image)
        s_metric = SegmentationMetrics(10)
        s_metric.update(pred_tta, gt)
        s_res = s_metric.compute()
        val_details.append({
            'miou': s_res['mIoU'],
            'image': img_raw,
            'gt': gt,
            'pred': pred_tta,
            'filename': filename
        })

    res_no_tta = metrics_no_tta.compute()
    res_tta    = metrics_tta.compute()

    print(f"\n[Results] Base mIoU: {res_no_tta['mIoU']*100:.2f}%")
    print(f"[Results] TTA  mIoU: {res_tta['mIoU']*100:.2f}%")
    print(f"[Results] Boost:    { (res_tta['mIoU'] - res_no_tta['mIoU'])*100:+.2f}%")
    print(f"[Performance] Avg inference time (TTA 4-aug): {np.mean(inference_times)*1000:.1f}ms / image")

    # 2. Save Best/Worst Examples
    val_details.sort(key=lambda x: x['miou'], reverse=True)
    
    print("[Report] Saving best/worst examples...")
    for rank in range(10):
        item = val_details[rank]
        save_comparison_grid(item['image'], item['gt'], item['pred'], 
                             final_dir / 'best_examples' / f"rank{rank+1:02d}_{item['filename']}",
                             f"IoU: {item['miou']*100:.2f}%")
    
    for rank in range(1, 6):
        item = val_details[-rank]
        save_comparison_grid(item['image'], item['gt'], item['pred'], 
                             final_dir / 'worst_examples' / f"worst{rank:02d}_{item['filename']}",
                             f"IoU: {item['miou']*100:.2f}%")

    # 3. Report Plots
    print("[Report] Generating charts...")
    # Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = res_tta['conf_matrix'].astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted'); plt.ylabel('Ground Truth'); plt.title('Confusion Matrix (Normalized) - TTA Model')
    plt.tight_layout()
    plt.savefig(final_dir / 'confusion_matrix.png', dpi=100)
    plt.close()

    # Comparison Chart
    plot_per_class_comparison(PHASE1_IOU, PHASE2_IOU, PHASE3_IOU, res_tta['iou_per_class'], final_dir / 'iou_chart.png')

    # 4. Results.txt
    summary_path = final_dir / 'results.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("OFFROAD SEGMENTATION FINAL REPORT\n")
        f.write("=================================\n\n")
        f.write(f"Base Model: DeepLabV3+ ResNet34 (512x512)\n")
        f.write(f"Best Local Checkpoint: runs/phase3/best_model.pth\n\n")
        
        f.write("MIOU SUMMARY:\n")
        f.write(f"---------------------------------\n")
        f.write(f"Phase 1: {0.5275*100:.2f}%\n")
        f.write(f"Phase 2: {0.5662*100:.2f}%\n")
        f.write(f"Phase 3: {0.5820*100:.2f}%\n")
        f.write(f"TTA Boost: {res_tta['mIoU']*100:.2f}%\n")
        f.write(f"Final Gain: {(res_tta['mIoU'] - 0.5275)*100:+.2f}%\n\n")
        
        f.write("PER-CLASS IOU PROGRESSION:\n")
        f.write(f"{'Class':<20} {'P1':>8} {'P2':>8} {'P3':>8} {'TTA':>8} {'Gain':>8}\n")
        f.write("-" * 65 + "\n")
        tta_iou = res_tta['iou_per_class']
        for i, name in enumerate(CLASS_NAMES):
            gain = (tta_iou[i] - PHASE1_IOU[i]) * 100
            f.write(f"{name:<20} {PHASE1_IOU[i]*100:>7.2f}% {PHASE2_IOU[i]*100:>7.2f}% {PHASE3_IOU[i]*100:>7.2f}% {tta_iou[i]*100:>7.2f}% {gain:>+7.2f}%\n")
        
        f.write("\nINFERENCE PERFORMANCE:\n")
        f.write(f"Average time per image (4-aug TTA): {np.mean(inference_times)*1000:.1f} ms\n")

    # 5. Final Test Submission
    test_ds = OffRoadDataset(test_images, transform=None, return_filename=True)
    print(f"\n[Test] Generating predictions for {len(test_ds)} images...")
    
    for i in tqdm(range(len(test_ds)), desc="Test Inference"):
        img_raw, filename = test_ds[i]
        
        # Inference
        input_tensor = val_norm_tfm(image=img_raw)['image'].unsqueeze(0).to(device)
        probs = tta_engine.predict(input_tensor)
        
        # Upsample to original resolution
        h_orig, w_orig = img_raw.shape[:2]
        probs_up = F.interpolate(probs, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
        pred = probs_up.argmax(1).cpu().numpy()[0]
        
        # Save colorized prediction
        pred_rgb = mask_to_rgb(pred)
        Image.fromarray(pred_rgb).save(final_dir / 'predictions' / filename)
        
        # Save overlay
        overlay = ((0.5 * img_raw + 0.5 * pred_rgb)).astype(np.uint8)
        Image.fromarray(overlay).save(final_dir / 'overlays' / filename)
        
        # (Optional) For actual submission you might need raw pixel IDs 100, 200...
        # We can also save the raw mask for competition script
        raw_mask = np.zeros_like(pred, dtype=np.int32)
        for cls_idx, raw_id in INV_CLASS_MAP.items():
            raw_mask[pred == cls_idx] = raw_id
        # We'll save this if needed, usually competitions want raw masks
        # Path(final_dir / 'raw_masks').mkdir(exist_ok=True)
        # Image.fromarray(raw_mask.astype(np.uint16)).save(final_dir / 'raw_masks' / filename)

    print(f"\n{'='*65}")
    print(f" ALL DONE! Final results and visuals saved to: {final_dir}")
    print(f" Best TTA mIoU achieved: {res_tta['mIoU']*100:.2f}%")
    print(f"{'='*65}")

if __name__ == '__main__':
    main()
