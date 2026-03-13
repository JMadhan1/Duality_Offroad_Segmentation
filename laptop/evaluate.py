"""
evaluate.py — Evaluation with detailed metrics on val or test set.

Usage:
    # Evaluate best checkpoint on val set
    python evaluate.py

    # Evaluate on test set (if ground-truth masks are available)
    python evaluate.py --split test

    # Evaluate pre-computed prediction masks (already saved PNGs)
    python evaluate.py --pred-masks runs/predictions/masks/

    # Skip inference and only re-compute metrics from saved masks
    python evaluate.py --pred-masks runs/predictions/masks/ --split test

Outputs:
    runs/eval/
        metrics.txt              — plain-text summary
        confusion_matrix.png     — normalised confusion matrix
        per_class_iou.png        — bar chart
        comparison_<n>.png       — side-by-side sample visualisations
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from src.utils      import (load_config, mask_to_rgb, save_comparison,
                             plot_confusion_matrix, plot_per_class_iou, CLASS_COLORS)
from src.dataset    import OffRoadDataset, remap_mask, CLASS_NAMES, NUM_CLASSES, IGNORE_INDEX
from src.transforms import get_val_transforms
from src.metrics    import SegmentationMetrics, print_metrics
from src.model      import build_model


# ─────────────────────────────────────────────────────────────────────────────
def run_model_inference(model, images_dir, cfg, device):
    """Run model inference and return dict: filename → pred_mask (np.uint8 [H,W])."""
    from src.model import multiscale_predict
    from src.transforms import get_test_transforms

    inf_cfg    = cfg.get('inference', {})
    input_size = tuple(cfg['training']['input_size'])
    tta_scales = inf_cfg.get('tta_scales', [0.75, 1.0, 1.25])
    use_tta    = inf_cfg.get('use_tta', True)
    use_amp    = cfg['training'].get('mixed_precision', True) and device.type == 'cuda'

    tfm = get_test_transforms(input_size)
    ds  = OffRoadDataset(images_dir=images_dir, transform=tfm, return_filename=True)

    preds = {}
    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(ds)), desc='Inference', dynamic_ncols=True):
            img_tensor, filename = ds[idx]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                if use_tta:
                    probs = multiscale_predict(model, img_tensor, scales=tta_scales)
                else:
                    logits = model(img_tensor)
                    probs  = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            preds[filename] = pred
    return preds


def load_pred_masks(pred_dir: str):
    """Load pre-saved prediction PNGs (class index 0-9)."""
    pred_path = Path(pred_dir)
    preds = {}
    for f in sorted(pred_path.glob('*.png')):
        mask = np.array(Image.open(f))
        preds[f.name] = mask
    return preds


def evaluate_predictions(
    preds: dict,
    masks_dir: str,
    output_dir: str,
    cfg: dict,
    n_comparisons: int = 8,
):
    """Compute metrics and save all plots."""
    from src.dataset import remap_mask
    import os

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    masks_path  = Path(masks_dir)
    num_classes = cfg['classes']['num_classes']

    metrics = SegmentationMetrics(num_classes, IGNORE_INDEX)
    comparison_count = 0

    filenames = sorted(preds.keys())
    for fname in tqdm(filenames, desc='Evaluating', dynamic_ncols=True):
        mask_file = masks_path / fname
        if not mask_file.exists():
            continue

        mask_raw = np.array(Image.open(mask_file))
        if mask_raw.ndim == 3:
            mask_raw = mask_raw[:, :, 0]
        gt_mask = remap_mask(mask_raw)

        pred_mask = preds[fname].astype(np.uint8)

        # Resize pred to gt if different
        if pred_mask.shape != gt_mask.shape:
            pred_pil = Image.fromarray(pred_mask).resize(
                (gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST
            )
            pred_mask = np.array(pred_pil)

        metrics.update(pred_mask, gt_mask)

        # Save comparison visualisations
        if comparison_count < n_comparisons:
            # Try to find the RGB image (adjust path as needed)
            img_candidates = [
                masks_path.parent / 'Color_Images' / fname,
                masks_path.parent / fname,
            ]
            orig_img = None
            for cand in img_candidates:
                if cand.exists():
                    orig_img = np.array(Image.open(cand).convert('RGB'))
                    break
            if orig_img is None:
                orig_img = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)

            save_comparison(
                image=orig_img,
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                save_path=str(out / f'comparison_{comparison_count:04d}.png'),
                class_names=CLASS_NAMES,
                title=fname,
            )
            comparison_count += 1

    results = metrics.compute()
    print_metrics(results, CLASS_NAMES)

    # ── Text report ────────────────────────────────────────────────────────
    report_path = out / 'metrics.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Offroad Segmentation — Evaluation Metrics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"mIoU:            {results['mIoU']*100:.2f}%\n")
        f.write(f"mDice:           {results['mDice']*100:.2f}%\n")
        f.write(f"Pixel Accuracy:  {results['pixel_accuracy']*100:.2f}%\n\n")
        f.write(f"{'Class':<20} {'IoU':>8} {'Dice':>8}\n")
        f.write("-" * 40 + "\n")
        for i, name in enumerate(CLASS_NAMES):
            iou  = results['iou_per_class'][i]
            dice = results['dice_per_class'][i]
            f.write(f"{name:<20} {iou*100:>7.2f}% {dice*100:>7.2f}%\n")
    print(f"[Report] {report_path}")

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_confusion_matrix(
        results['conf_matrix'],
        CLASS_NAMES,
        save_path=str(out / 'confusion_matrix.png'),
    )
    plot_per_class_iou(
        results['iou_per_class'],
        CLASS_NAMES,
        save_path=str(out / 'per_class_iou.png'),
    )

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation model')
    parser.add_argument('--config',       default='config.yaml')
    parser.add_argument('--split',        default='val', choices=['val', 'test'],
                        help='Which data split to evaluate on')
    parser.add_argument('--checkpoint',   default=None)
    parser.add_argument('--pred-masks',   default=None,
                        help='Directory of pre-computed prediction PNGs (skip inference)')
    parser.add_argument('--output',       default='runs/eval')
    parser.add_argument('--n-samples',    default=8, type=int,
                        help='Number of comparison images to save')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Data split ─────────────────────────────────────────────────────────
    if args.split == 'val':
        images_dir = cfg['data']['val_images']
        masks_dir  = cfg['data']['val_masks']
    else:
        images_dir = cfg['data']['test_images']
        masks_dir  = cfg['data'].get('test_masks', '')
    print(f"[Eval]   split={args.split}  images={images_dir}")

    # ── Get predictions ─────────────────────────────────────────────────────
    if args.pred_masks:
        print(f"[Preds]  Loading from {args.pred_masks}")
        preds = load_pred_masks(args.pred_masks)
    else:
        runs_dir  = Path(cfg['output']['runs_dir'])
        ckpt_path = args.checkpoint
        if not ckpt_path:
            ema_path = runs_dir / cfg['output'].get('ema_checkpoint_name', 'best_model_ema.pth')
            reg_path = runs_dir / cfg['output']['checkpoint_name']
            ckpt_path = str(ema_path) if ema_path.exists() else str(reg_path)
        print(f"[Ckpt]   {ckpt_path}")

        model = build_model(cfg).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        preds = run_model_inference(model, images_dir, cfg, device)

    # ── Evaluate ────────────────────────────────────────────────────────────
    if not masks_dir or not Path(masks_dir).exists():
        print("[Warning] No ground-truth masks directory found. Saving predictions only.")
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        for fname, pred in preds.items():
            from src.utils import mask_to_rgb
            color = mask_to_rgb(pred)
            Image.fromarray(color).save(str(out / fname))
        print(f"[Output] Saved colour masks to {out}/")
        return

    evaluate_predictions(preds, masks_dir, args.output, cfg, n_comparisons=args.n_samples)
    print(f"\n[Done] All outputs saved to {args.output}/")


if __name__ == '__main__':
    main()
