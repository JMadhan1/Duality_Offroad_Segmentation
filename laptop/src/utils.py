"""
utils.py — Utilities: EMA, checkpointing, visualization, config loading.
"""

import os
import csv
import copy
import yaml
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')   # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str = 'config.yaml') -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Colour palette (high-contrast)
# --------------------------------------------------------------------------- #
CLASS_COLORS = {
    0: (34,  139,  34),    # Trees          — Forest Green
    1: (50,  205,  50),    # Lush Bushes    — Lime Green
    2: (189, 183, 107),    # Dry Grass      — Dark Khaki
    3: (139, 119,  42),    # Dry Bushes     — Dark Goldenrod
    4: (160,  82,  45),    # Ground Clutter — Sienna
    5: (255,   0, 255),    # Flowers        — Magenta
    6: (139,  69,  19),    # Logs           — Saddle Brown
    7: (128, 128, 128),    # Rocks          — Gray
    8: (210, 180, 140),    # Landscape      — Tan
    9: (135, 206, 235),    # Sky            — Sky Blue
}
IGNORE_COLOR = (0, 0, 0)   # Black for ignore_index pixels


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert class-index mask [H, W] → RGB image [H, W, 3]."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_idx, color in CLASS_COLORS.items():
        rgb[mask == cls_idx] = color
    return rgb


# --------------------------------------------------------------------------- #
# EMA — Exponential Moving Average of model weights
# --------------------------------------------------------------------------- #
class EMA:
    """Shadow-copy EMA of model parameters.

    Usage:
        ema = EMA(model, decay=0.9999)
        # inside training loop:
        ema.update()
        # at evaluation:
        ema.apply_shadow()
        val_miou = evaluate(model, ...)
        ema.restore()
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model  = model
        self.decay  = decay
        self.shadow: dict = {}
        self.backup: dict = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self):
        """Temporarily replace model weights with EMA weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup.clear()

    def state_dict(self) -> dict:
        return {'shadow': self.shadow, 'decay': self.decay}

    def load_state_dict(self, sd: dict):
        self.shadow = sd['shadow']
        self.decay  = sd['decay']


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    miou: float,
    ema: Optional[EMA] = None,
    extra: dict | None = None,
):
    ckpt = {
        'epoch':      epoch,
        'miou':       miou,
        'model':      model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'scheduler':  scheduler.state_dict() if scheduler is not None else None,
    }
    if ema is not None:
        ckpt['ema'] = ema.state_dict()
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    ema: Optional[EMA] = None,
    device: str = 'cpu',
) -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and ckpt.get('scheduler') is not None:
        scheduler.load_state_dict(ckpt['scheduler'])
    if ema is not None and 'ema' in ckpt:
        ema.load_state_dict(ckpt['ema'])
    print(f"[Checkpoint] Loaded from {path}  (epoch={ckpt['epoch']}, mIoU={ckpt['miou']:.4f})")
    return ckpt


# --------------------------------------------------------------------------- #
# CSV Logger
# --------------------------------------------------------------------------- #
class CSVLogger:
    def __init__(self, path: str):
        self.path = path
        self._header_written = Path(path).exists()
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: dict):
        write_header = not self._header_written
        with open(self.path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


# --------------------------------------------------------------------------- #
# Visualisation helpers
# --------------------------------------------------------------------------- #
def make_legend_patches(class_names: list) -> list:
    return [
        mpatches.Patch(color=np.array(CLASS_COLORS[i]) / 255., label=name)
        for i, name in enumerate(class_names)
    ]


def save_comparison(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    save_path: str,
    class_names: list,
    title: str = '',
):
    """Save side-by-side: RGB | Ground Truth | Prediction | Error map."""
    gt_rgb   = mask_to_rgb(gt_mask)
    pred_rgb = mask_to_rgb(pred_mask)

    # Error map: highlight misclassified pixels in red
    error = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    valid = (gt_mask != 255)
    error[valid & (gt_mask != pred_mask)] = (255, 0, 0)
    error[valid & (gt_mask == pred_mask)] = (0, 200, 0)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, img, ttl in zip(
        axes,
        [image, gt_rgb, pred_rgb, error],
        ['RGB Image', 'Ground Truth', 'Prediction', 'Error (red=wrong)'],
    ):
        ax.imshow(img)
        ax.set_title(ttl, fontsize=11)
        ax.axis('off')

    if class_names:
        patches = make_legend_patches(class_names)
        fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=8,
                   bbox_to_anchor=(0.5, -0.12))

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close(fig)


def save_overlay(
    image: np.ndarray,
    pred_mask: np.ndarray,
    save_path: str,
    alpha: float = 0.45,
):
    """Save RGB image with semi-transparent prediction overlay."""
    pred_rgb = mask_to_rgb(pred_mask).astype(np.float32)
    img_f    = image.astype(np.float32)
    blended  = ((1 - alpha) * img_f + alpha * pred_rgb).clip(0, 255).astype(np.uint8)
    Image.fromarray(blended).save(save_path)


def plot_training_curves(log_csv: str, out_dir: str):
    """Plot loss and mIoU curves from CSV log."""
    import pandas as pd
    df = pd.read_csv(log_csv)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Loss curves
    fig, ax = plt.subplots(figsize=(10, 5))
    if 'train_loss' in df.columns:
        ax.plot(df['epoch'], df['train_loss'], label='Train Loss')
    if 'val_loss' in df.columns:
        ax.plot(df['epoch'], df['val_loss'], label='Val Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss'); ax.legend(); ax.grid(True)
    fig.savefig(os.path.join(out_dir, 'loss_curve.png'), bbox_inches='tight')
    plt.close(fig)

    # mIoU curve
    if 'val_miou' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['epoch'], df['val_miou'] * 100, color='green', label='Val mIoU')
        ax.set_xlabel('Epoch'); ax.set_ylabel('mIoU (%)')
        ax.set_title('Validation mIoU over Training'); ax.legend(); ax.grid(True)
        ax.set_ylim(0, 100)
        fig.savefig(os.path.join(out_dir, 'miou_curve.png'), bbox_inches='tight')
        plt.close(fig)

    print(f"[Plots] Saved training curves to {out_dir}/")


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: list,
    save_path: str,
    normalize: bool = True,
):
    import seaborn as sns
    cm = conf_matrix.astype(np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.where(row_sums > 0, cm / row_sums, 0.0)
        fmt, vmax = '.2f', 1.0
    else:
        fmt, vmax = 'd', None

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        vmin=0, vmax=vmax, ax=ax,
    )
    ax.set_xlabel('Predicted Class'); ax.set_ylabel('Ground Truth Class')
    ax.set_title('Confusion Matrix (row-normalized)' if normalize else 'Confusion Matrix')
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close(fig)
    print(f"[Plot] Saved confusion matrix → {save_path}")


def plot_per_class_iou(iou_per_class: list, class_names: list, save_path: str):
    """Bar chart of per-class IoU."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [np.array(CLASS_COLORS[i]) / 255. for i in range(len(class_names))]
    bars = ax.bar(class_names, [v * 100 for v in iou_per_class], color=colors, edgecolor='black')
    ax.axhline(y=np.mean(iou_per_class) * 100, color='red', linestyle='--', label=f'mIoU={np.mean(iou_per_class)*100:.1f}%')
    for bar, val in zip(bars, iou_per_class):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Class'); ax.set_ylabel('IoU (%)')
    ax.set_title('Per-Class IoU'); ax.set_ylim(0, 105)
    ax.legend(); ax.grid(axis='y', alpha=0.4)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close(fig)
    print(f"[Plot] Saved per-class IoU → {save_path}")


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
