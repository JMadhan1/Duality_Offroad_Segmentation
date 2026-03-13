"""
verify_data.py — Data pipeline verification. Run this BEFORE training.

Checks:
  1. Raw mask pixel values (should be 100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000)
  2. Remapped mask values (should be 0-9 only)
  3. Pixel counts per class across 5 random train samples
  4. Class weights tensor (no zeros, NaNs, Infs; max < 50)
  5. Saves RGB | raw-mask | remapped-mask visualisations
  6. Verifies a full DataLoader batch is healthy
  7. Checks untrained model prediction distribution (~10% per class expected)

Usage:
    python verify_data.py
"""

import sys
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.utils.data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))
from src.dataset    import OffRoadDataset, remap_mask, CLASS_MAP, CLASS_NAMES, NUM_CLASSES, IGNORE_INDEX
from src.transforms import get_val_transforms
from src.utils      import load_config

# ─── Colours ──────────────────────────────────────────────────────────────────
RAW_COLORS = {
    100:   (34,  139,  34),
    200:   (50,  205,  50),
    300:   (189, 183, 107),
    500:   (139, 119,  42),
    550:   (160,  82,  45),
    600:   (255,   0, 255),
    700:   (139,  69,  19),
    800:   (128, 128, 128),
    7100:  (210, 180, 140),
    10000: (135, 206, 235),
    0:     (255,   0,   0),   # 0 values -> red (unexpected)
}
REMAP_COLORS = {
    0: (34, 139, 34),  1: (50, 205, 50),  2: (189, 183, 107),
    3: (139, 119, 42), 4: (160, 82, 45),  5: (255, 0, 255),
    6: (139, 69, 19),  7: (128, 128, 128), 8: (210, 180, 140),
    9: (135, 206, 235),
}


def colorize_raw(mask_raw):
    out = np.zeros((*mask_raw.shape, 3), dtype=np.uint8)
    for val, color in RAW_COLORS.items():
        out[mask_raw == val] = color
    unknown = ~np.isin(mask_raw, list(RAW_COLORS.keys()))
    out[unknown] = (255, 165, 0)   # orange = unknown
    return out


def colorize_remapped(mask):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_idx, color in REMAP_COLORS.items():
        out[mask == cls_idx] = color
    out[mask == IGNORE_INDEX] = (255, 0, 0)   # ignore -> red
    return out


def main():
    cfg       = load_config('config.yaml')
    train_img = cfg['data']['train_images']
    train_msk = cfg['data']['train_masks']
    out_dir   = Path('runs/verify')
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  DATA PIPELINE VERIFICATION")
    print("=" * 65)

    # ── 1. CLASS MAP ────────────────────────────────────────────────────────
    print("\n[1] CLASS MAP (raw pixel ID -> class index):")
    for raw_id, cls_idx in sorted(CLASS_MAP.items()):
        print(f"     raw {raw_id:>5} -> [{cls_idx}] {CLASS_NAMES[cls_idx]}")

    # ── 2. FILE COUNT ────────────────────────────────────────────────────────
    img_files  = sorted(Path(train_img).glob('*.png'))
    mask_files = sorted(Path(train_msk).glob('*.png'))
    print(f"\n[2] Files:  images={len(img_files)}  masks={len(mask_files)}")
    assert len(img_files) > 0,  f"No images in {train_img}"
    assert len(mask_files) > 0, f"No masks in {train_msk}"

    # Name alignment check
    img_names  = {f.name for f in img_files}
    mask_names = {f.name for f in mask_files}
    missing_masks = img_names - mask_names
    if missing_masks:
        print(f"  [!] {len(missing_masks)} images have no mask, e.g.: {list(missing_masks)[:3]}")
    else:
        print("  Name alignment: OK OK")

    # ── 3. SAMPLE INSPECTION ─────────────────────────────────────────────────
    print("\n[3] SAMPLE INSPECTION (5 random images):")
    random.seed(42)
    indices = random.sample(range(len(img_files)), min(5, len(img_files)))
    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    all_good = True

    for si, idx in enumerate(indices):
        img_p  = img_files[idx]
        msk_p  = mask_files[idx]

        img_arr  = np.array(Image.open(img_p).convert('RGB'))
        msk_raw  = np.array(Image.open(msk_p))
        if msk_raw.ndim == 3:
            msk_raw = msk_raw[:, :, 0]
        msk_remap = remap_mask(msk_raw)

        raw_unique   = sorted(np.unique(msk_raw).tolist())
        remap_unique = sorted(np.unique(msk_remap).tolist())

        unexpected = [v for v in raw_unique if v not in CLASS_MAP]
        flag = f"  [!] UNEXPECTED RAW VALUES: {unexpected}" if unexpected else "  OK"
        if unexpected:
            all_good = False

        bad_remap = [v for v in remap_unique if v not in list(range(NUM_CLASSES)) + [IGNORE_INDEX]]
        if bad_remap:
            flag += f"  [!] BAD REMAP: {bad_remap}"
            all_good = False

        print(f"  [{si}] {img_p.name}  raw={raw_unique}  remapped={remap_unique}{flag}")

        for c in range(NUM_CLASSES):
            pixel_counts[c] += (msk_remap == c).sum()

        # Save visualisation for first 3 samples
        if si < 3:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].imshow(img_arr);               axes[0].set_title('RGB Image');      axes[0].axis('off')
            axes[1].imshow(colorize_raw(msk_raw)); axes[1].set_title('Raw Mask');       axes[1].axis('off')
            axes[2].imshow(colorize_remapped(msk_remap)); axes[2].set_title('Remapped (0-9)'); axes[2].axis('off')
            patches = [mpatches.Patch(color=np.array(REMAP_COLORS[i])/255., label=f"[{i}]{CLASS_NAMES[i]}")
                       for i in range(NUM_CLASSES)]
            fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=7,
                       bbox_to_anchor=(0.5, -0.05))
            plt.suptitle(img_p.name); plt.tight_layout()
            fig.savefig(out_dir / f'sample_{si:02d}.png', bbox_inches='tight', dpi=80)
            plt.close(fig)

    # ── 4. PIXEL DISTRIBUTION ────────────────────────────────────────────────
    print("\n[4] PIXEL DISTRIBUTION (5 samples):")
    total = pixel_counts.sum()
    for c in range(NUM_CLASSES):
        pct = 100.0 * pixel_counts[c] / max(total, 1)
        print(f"  [{c}] {CLASS_NAMES[c]:<18} {pixel_counts[c]:>10,}  ({pct:5.1f}%)  {'#' * max(1, int(pct/2))}")

    zero_cls = [CLASS_NAMES[c] for c in range(NUM_CLASSES) if pixel_counts[c] == 0]
    if zero_cls:
        print(f"\n  [!] Zero pixels in these 5 samples: {zero_cls}")

    # ── 5. CLASS WEIGHTS ─────────────────────────────────────────────────────
    print("\n[5] CLASS WEIGHTS:")
    wfile = Path('runs/class_weights.pt')
    if wfile.exists():
        cw = torch.load(wfile, weights_only=True)
        print(f"  Loaded from {wfile}")
    else:
        print("  class_weights.pt not found — computing from 5-sample counts ...")
        inv = np.where(pixel_counts > 0, float(total) / (NUM_CLASSES * pixel_counts.astype(float)), 0.0)
        if inv.mean() > 0:
            inv = inv / inv.mean()
        cw = torch.tensor(inv, dtype=torch.float32)

    ok = True
    for c in range(NUM_CLASSES):
        w = cw[c].item()
        issues = []
        if w == 0:           issues.append("ZERO")
        if np.isnan(w):      issues.append("NaN")
        if np.isinf(w):      issues.append("Inf")
        if w > 50:           issues.append(f"VERY LARGE")
        flag = f"  [!] {'/'.join(issues)}" if issues else ""
        if issues: ok = False
        print(f"  [{c}] {CLASS_NAMES[c]:<18}  weight={w:8.4f}{flag}")

    print(f"\n  Weights: {'ALL OK OK' if ok else '[!] Issues detected!'}")

    # ── 6. DATALOADER BATCH ──────────────────────────────────────────────────
    print("\n[6] DATALOADER BATCH:")
    ds  = OffRoadDataset(train_img, train_msk, transform=get_val_transforms(tuple(cfg['training']['input_size'])))
    dl  = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    imgs_b, masks_b = next(iter(dl))
    print(f"  images: {imgs_b.shape}  range=[{imgs_b.min():.2f}, {imgs_b.max():.2f}]")
    print(f"  masks:  {masks_b.shape}  unique={masks_b.unique().tolist()}")
    ign_frac = (masks_b == IGNORE_INDEX).float().mean().item()
    if ign_frac > 0.5:
        print(f"  [!] {ign_frac*100:.1f}% pixels are IGNORE_INDEX — remapping is BROKEN!")
        all_good = False
    else:
        print(f"  ignore pixels: {ign_frac*100:.2f}%  OK")

    # ── 7. UNTRAINED MODEL FORWARD PASS ─────────────────────────────────────
    print("\n[7] UNTRAINED MODEL PREDICTION DISTRIBUTION:")
    print("    (Should be ~10% per class for an unbiased model)")
    from src.model import build_model
    model = build_model(cfg)
    model.eval()
    with torch.no_grad():
        logits = model(imgs_b)
    preds = logits.argmax(dim=1)
    print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    for c in range(NUM_CLASSES):
        pct = (preds == c).float().mean().item() * 100
        bar = '#' * max(1, int(pct / 2))
        flag = "  [!] DOMINATING" if pct > 60 else ""
        print(f"  [{c}] {CLASS_NAMES[c]:<18}  {pct:5.1f}%  {bar}{flag}")

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if all_good and ign_frac < 0.5:
        print("  RESULT: Data pipeline CORRECT OK  — safe to train")
    else:
        print("  RESULT: [!] Issues detected — fix before training!")
    print(f"  Visualisations -> {out_dir}/")
    print("=" * 65)


if __name__ == '__main__':
    main()
