"""
metrics.py — IoU, mIoU, confusion matrix, pixel accuracy.

All functions operate on NumPy arrays (class predictions and targets).
ignore_index pixels are excluded from all computations.
"""

import numpy as np
import torch
from typing import Optional


# --------------------------------------------------------------------------- #
# Core accumulator
# --------------------------------------------------------------------------- #
class SegmentationMetrics:
    """Running accumulator for segmentation metrics.

    Usage:
        metrics = SegmentationMetrics(num_classes=10)
        for preds, targets in loader:
            metrics.update(preds, targets)
        results = metrics.compute()
        metrics.reset()
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        # Confusion matrix: rows = gt class, cols = pred class
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.pixel_correct = 0
        self.pixel_total   = 0

    def update(
        self,
        preds: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
    ):
        """Add one batch or single image.

        Args:
            preds:   predicted class indices [H, W] or [B, H, W], int.
            targets: ground-truth class indices [H, W] or [B, H, W], int.
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        preds   = preds.flatten().astype(np.int64)
        targets = targets.flatten().astype(np.int64)

        # Mask out ignored pixels
        valid = targets != self.ignore_index
        preds   = preds[valid]
        targets = targets[valid]

        # Pixel accuracy
        self.pixel_correct += (preds == targets).sum()
        self.pixel_total   += valid.sum()

        # Accumulate confusion matrix using bincount trick
        combined = targets * self.num_classes + preds
        cm_flat  = np.bincount(combined, minlength=self.num_classes ** 2)
        self.conf_matrix += cm_flat.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        """Return dict with mIoU, per-class IoU, dice, pixel accuracy."""
        cm = self.conf_matrix
        # TP[c] = cm[c, c]
        # FP[c] = sum(col c) - cm[c, c]   (predicted c but not gt c)
        # FN[c] = sum(row c) - cm[c, c]   (gt c but not predicted c)
        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        iou_per_class = tp / (tp + fp + fn + 1e-10)
        # Only average over classes that actually appear in targets
        present = (cm.sum(axis=1) > 0)
        miou    = iou_per_class[present].mean() if present.any() else 0.0

        # Dice = 2TP / (2TP + FP + FN)
        dice_per_class = 2 * tp / (2 * tp + fp + fn + 1e-10)
        mdice = dice_per_class[present].mean() if present.any() else 0.0

        pixel_acc = (
            self.pixel_correct / self.pixel_total
            if self.pixel_total > 0 else 0.0
        )

        return {
            'mIoU':           float(miou),
            'mDice':          float(mdice),
            'pixel_accuracy': float(pixel_acc),
            'iou_per_class':  iou_per_class.tolist(),
            'dice_per_class': dice_per_class.tolist(),
            'conf_matrix':    cm,
            'present_classes': present.tolist(),
        }


# --------------------------------------------------------------------------- #
# Convenience: compute IoU for a single batch (no accumulation)
# --------------------------------------------------------------------------- #
def batch_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> tuple[float, np.ndarray]:
    """Quick IoU computation for a single batch (training logging).

    Returns (mIoU, iou_per_class_array).
    """
    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
    metrics.update(preds, targets)
    results = metrics.compute()
    return results['mIoU'], np.array(results['iou_per_class'])


# --------------------------------------------------------------------------- #
# Pretty printer
# --------------------------------------------------------------------------- #
def print_metrics(results: dict, class_names: list[str], epoch: Optional[int] = None):
    """Print formatted metrics table."""
    prefix = f"[Epoch {epoch}] " if epoch is not None else ""
    print(f"\n{prefix}{'='*65}")
    print(f"  mIoU: {results['mIoU']*100:.2f}%   mDice: {results['mDice']*100:.2f}%   "
          f"PixelAcc: {results['pixel_accuracy']*100:.2f}%")
    print(f"  {'Class':<20} {'IoU':>8} {'Dice':>8}")
    print(f"  {'-'*40}")
    iou  = results['iou_per_class']
    dice = results['dice_per_class']
    for i, name in enumerate(class_names):
        flag = "✓" if results['present_classes'][i] else "-"
        print(f"  {flag} {name:<18} {iou[i]*100:>7.2f}% {dice[i]*100:>7.2f}%")
    print(f"{'='*65}\n")
