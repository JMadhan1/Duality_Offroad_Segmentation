"""
visualize.py — High-quality visualisation of segmentation results.

Usage:
    # Visualise ground-truth masks from val set
    python visualize.py --split val --n 20

    # Visualise predictions vs ground truth
    python visualize.py --pred-masks runs/predictions/masks/ --split val

    # Visualise test-set predictions (no GT, overlay only)
    python visualize.py --pred-masks runs/predictions/masks/ --split test

Outputs written to runs/visualizations/
"""

import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))
from src.utils   import load_config, mask_to_rgb, CLASS_COLORS, save_comparison, save_overlay
from src.dataset import remap_mask, CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# Legend helper
# ─────────────────────────────────────────────────────────────────────────────
def legend_patches():
    return [
        mpatches.Patch(
            color=np.array(CLASS_COLORS[i]) / 255.,
            label=f"[{i}] {CLASS_NAMES[i]}"
        )
        for i in range(NUM_CLASSES)
    ]


def save_gt_grid(images_dir, masks_dir, output_dir, n_samples=16):
    """Grid showing RGB images with their ground-truth masks."""
    img_paths  = sorted(Path(images_dir).glob('*.png'))[:n_samples]
    mask_paths = [Path(masks_dir) / p.name for p in img_paths]

    cols = 4
    rows = max(1, (len(img_paths) * 2 + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    ax_idx = 0
    for img_p, msk_p in zip(img_paths, mask_paths):
        if not msk_p.exists():
            continue
        img      = np.array(Image.open(img_p).convert('RGB'))
        mask_raw = np.array(Image.open(msk_p))
        if mask_raw.ndim == 3:
            mask_raw = mask_raw[:, :, 0]
        gt   = remap_mask(mask_raw)
        gt_c = mask_to_rgb(gt)

        if ax_idx + 1 < len(axes):
            axes[ax_idx].imshow(img);   axes[ax_idx].set_title(img_p.name[:20], fontsize=7); axes[ax_idx].axis('off')
            axes[ax_idx+1].imshow(gt_c); axes[ax_idx+1].set_title('GT Mask', fontsize=7);      axes[ax_idx+1].axis('off')
        ax_idx += 2

    for a in axes[ax_idx:]:
        a.axis('off')

    fig.legend(handles=legend_patches(), loc='lower center', ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, -0.05))
    plt.suptitle('Ground-Truth Segmentation', fontsize=14, y=1.01)
    plt.tight_layout()
    out = Path(output_dir) / 'gt_grid.png'
    fig.savefig(out, bbox_inches='tight', dpi=100)
    plt.close(fig)
    print(f"[Viz] GT grid saved → {out}")


def save_pred_grid(pred_dir, images_dir, output_dir, n_samples=16):
    """Grid of prediction colour masks alongside RGB images."""
    pred_paths = sorted(Path(pred_dir).glob('*.png'))[:n_samples]
    cols = 4
    rows = max(1, (len(pred_paths) * 2 + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    ax_idx = 0
    for pred_p in pred_paths:
        img_p = Path(images_dir) / pred_p.name
        if not img_p.exists():
            continue
        img  = np.array(Image.open(img_p).convert('RGB'))
        pred = np.array(Image.open(pred_p))
        pred_c = mask_to_rgb(pred)

        if ax_idx + 1 < len(axes):
            axes[ax_idx].imshow(img);    axes[ax_idx].set_title(pred_p.name[:20], fontsize=7); axes[ax_idx].axis('off')
            axes[ax_idx+1].imshow(pred_c); axes[ax_idx+1].set_title('Prediction', fontsize=7);   axes[ax_idx+1].axis('off')
        ax_idx += 2

    for a in axes[ax_idx:]:
        a.axis('off')

    fig.legend(handles=legend_patches(), loc='lower center', ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, -0.05))
    plt.suptitle('Model Predictions', fontsize=14, y=1.01)
    plt.tight_layout()
    out = Path(output_dir) / 'pred_grid.png'
    fig.savefig(out, bbox_inches='tight', dpi=100)
    plt.close(fig)
    print(f"[Viz] Pred grid saved → {out}")


def save_overlays_batch(pred_dir, images_dir, output_dir, n_samples=20, alpha=0.45):
    """Batch overlay: RGB + semi-transparent prediction."""
    pred_paths = sorted(Path(pred_dir).glob('*.png'))[:n_samples]
    out = Path(output_dir) / 'overlays'
    out.mkdir(parents=True, exist_ok=True)
    for pred_p in pred_paths:
        img_p = Path(images_dir) / pred_p.name
        if not img_p.exists():
            continue
        img  = np.array(Image.open(img_p).convert('RGB'))
        pred = np.array(Image.open(pred_p))
        save_overlay(img, pred, str(out / pred_p.name), alpha=alpha)
    print(f"[Viz] {len(pred_paths)} overlays saved → {out}/")


def save_full_comparisons(pred_dir, images_dir, masks_dir, output_dir, n_samples=10):
    """4-panel comparison: RGB | GT | Pred | Error map."""
    pred_paths = sorted(Path(pred_dir).glob('*.png'))[:n_samples]
    out = Path(output_dir) / 'comparisons'
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for pred_p in pred_paths:
        img_p  = Path(images_dir) / pred_p.name
        mask_p = Path(masks_dir)  / pred_p.name
        if not img_p.exists() or not mask_p.exists():
            continue
        img      = np.array(Image.open(img_p).convert('RGB'))
        mask_raw = np.array(Image.open(mask_p))
        if mask_raw.ndim == 3:
            mask_raw = mask_raw[:, :, 0]
        gt   = remap_mask(mask_raw)
        pred = np.array(Image.open(pred_p))

        if pred.shape != gt.shape:
            pred = np.array(
                Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
            )

        save_comparison(img, gt, pred, str(out / pred_p.name), CLASS_NAMES, title=pred_p.name)
        count += 1
    print(f"[Viz] {count} 4-panel comparisons saved → {out}/")


def main():
    parser = argparse.ArgumentParser(description='Visualise segmentation results')
    parser.add_argument('--config',     default='config.yaml')
    parser.add_argument('--split',      default='val', choices=['val', 'test'])
    parser.add_argument('--pred-masks', default=None,
                        help='Directory of prediction PNGs (class indices 0-9)')
    parser.add_argument('--output',     default='runs/visualizations')
    parser.add_argument('--n',          default=20, type=int, help='Number of samples to visualise')
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.split == 'val':
        images_dir = cfg['data']['val_images']
        masks_dir  = cfg['data']['val_masks']
    else:
        images_dir = cfg['data']['test_images']
        masks_dir  = cfg['data'].get('test_masks', '')

    print(f"[Viz] split={args.split}  n={args.n}  output={out}")

    # GT grid (always available for val)
    if masks_dir and Path(masks_dir).exists():
        save_gt_grid(images_dir, masks_dir, str(out), n_samples=args.n)

    # Prediction visualisations
    if args.pred_masks and Path(args.pred_masks).exists():
        save_pred_grid(args.pred_masks, images_dir, str(out), n_samples=args.n)
        save_overlays_batch(args.pred_masks, images_dir, str(out), n_samples=args.n)

        if masks_dir and Path(masks_dir).exists():
            save_full_comparisons(
                args.pred_masks, images_dir, masks_dir,
                str(out), n_samples=min(args.n, 10)
            )

    print(f"\n[Done] All visualisations saved to {out}/")


if __name__ == '__main__':
    main()
