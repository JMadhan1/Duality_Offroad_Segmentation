"""
phase2_train.py - Phase 2 fine-tuning for minority class improvement.

Loads Phase 1 best checkpoint and fine-tunes with:
  - WeightedRandomSampler (oversample images containing rare classes)
  - Combined CE + Dice loss with per-class weight caps
  - Low LR (5e-5) with CosineAnnealingLR

Run:
    python phase2_train.py
"""

import sys
import os
import time
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import yaml

# Local src imports
sys.path.insert(0, str(Path(__file__).parent))
from src.dataset import OffRoadDataset, CLASS_NAMES, IGNORE_INDEX, remap_mask
from src.model import build_deeplabv3plus
from src.metrics import SegmentationMetrics
from src.utils import set_seed, CSVLogger

# --------------------------------------------------------------------------- #
# Phase 1 baseline results (for final comparison table)
# --------------------------------------------------------------------------- #
PHASE1_IOU = [0.7713, 0.6466, 0.6379, 0.3540, 0.3156,
              0.5375, 0.2190, 0.2985, 0.5236, 0.9707]
PHASE1_MIOU = 0.5275

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
P2_CFG = {
    'checkpoint':  'runs/best_model.pth',
    'save_dir':    'runs/phase2',
    'epochs':       60,
    'learning_rate': 5e-5,
    'weight_decay':  0.01,
    'batch_size':    8,
    'input_size':   (384, 384),
    'mixed_precision': True,
    'num_workers':   0,
    'pin_memory':    True,
    'early_stopping_patience': 20,
    'num_classes':   10,
    'backbone':      'resnet34',
    'seed':          42,
}

# Rare class indices (zero-based)
_LOGS           = 6
_ROCKS          = 7
_GROUND_CLUTTER = 4
_DRY_BUSHES     = 3
RARE_CLASS_WEIGHTS = {_LOGS: 4.0, _ROCKS: 3.0, _GROUND_CLUTTER: 2.5, _DRY_BUSHES: 2.0}
RARE_PIXEL_THRESHOLD = 100   # min pixels for a class to be "present"

# Per-class weight caps (higher for rare classes, normalised after capping)
WEIGHT_CAPS = {0: 3.0, 1: 3.0, 2: 3.0, 3: 4.0, 4: 4.0,
               5: 3.0, 6: 8.0, 7: 5.0, 8: 3.0, 9: 3.0}

# --------------------------------------------------------------------------- #
# Transforms  (albumentations 2.0.8 API)
# --------------------------------------------------------------------------- #
def get_train_transform(h=384, w=384):
    return A.Compose([
        A.Resize(height=h, width=w),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(16, 32),
            hole_width_range=(16, 32),
            fill=0,
            fill_mask=255,
            p=0.3,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform(h=384, w=384):
    return A.Compose([
        A.Resize(height=h, width=w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# --------------------------------------------------------------------------- #
# WeightedRandomSampler helpers
# --------------------------------------------------------------------------- #
def compute_sample_weights(masks_dir: str, image_files: list) -> list:
    """Scan every training mask and assign a per-image sampling weight."""
    masks_path = Path(masks_dir)
    rare_counts = {c: 0 for c in RARE_CLASS_WEIGHTS}
    weights = []

    print(f"[Sampler] Scanning {len(image_files)} masks for rare classes ...")
    for img_path in tqdm(image_files, desc='Scanning masks', ncols=80):
        mask_file = masks_path / Path(img_path).name
        if not mask_file.exists():
            weights.append(1.0)
            continue
        mask_raw = np.array(Image.open(mask_file))
        if mask_raw.ndim == 3:
            mask_raw = mask_raw[:, :, 0]
        mask = remap_mask(mask_raw)

        img_weight = 1.0
        for cls_idx, cls_weight in RARE_CLASS_WEIGHTS.items():
            if (mask == cls_idx).sum() > RARE_PIXEL_THRESHOLD:
                rare_counts[cls_idx] += 1
                img_weight = max(img_weight, cls_weight)
        weights.append(img_weight)

    print("[Sampler] Images containing each rare class:")
    for cls_idx, count in rare_counts.items():
        print(f"  [{cls_idx}] {CLASS_NAMES[cls_idx]:<20} : {count:>4} images  "
              f"(weight={RARE_CLASS_WEIGHTS[cls_idx]})")

    weight_vals = sorted(set(weights))
    for w in weight_vals:
        n = sum(1 for x in weights if x == w)
        print(f"  weight={w:.1f} -> {n} images ({100*n/len(weights):.1f}%)")

    return weights


# --------------------------------------------------------------------------- #
# Class weights with per-class caps
# --------------------------------------------------------------------------- #
def compute_class_weights_capped(masks_dir: str, num_classes: int = 10) -> torch.Tensor:
    """Inverse-frequency weights with individual per-class caps, normalised to mean=1."""
    masks_path = Path(masks_dir)
    pixel_counts = np.zeros(num_classes, dtype=np.float64)

    mask_files = sorted(f for f in masks_path.iterdir() if f.suffix.lower() == '.png')
    print(f"[Weights] Computing from {len(mask_files)} masks ...")
    for mf in tqdm(mask_files, desc='Class weights', ncols=80):
        mask_raw = np.array(Image.open(mf))
        if mask_raw.ndim == 3:
            mask_raw = mask_raw[:, :, 0]
        remapped = remap_mask(mask_raw)
        for c in range(num_classes):
            pixel_counts[c] += (remapped == c).sum()

    total = pixel_counts.sum()
    weights = np.where(
        pixel_counts > 0,
        total / (num_classes * pixel_counts),
        0.0,
    )
    # Normalize to mean=1 before capping
    if weights.sum() > 0:
        weights = weights / weights.mean()

    # Apply per-class caps
    for c, cap in WEIGHT_CAPS.items():
        weights[c] = min(weights[c], cap)

    # Re-normalize to mean=1 after capping
    if weights.sum() > 0:
        weights = weights / weights.mean()

    print("[Weights] Final class weights (capped + normalized):")
    for i, (name, w) in enumerate(zip(CLASS_NAMES, weights)):
        print(f"  [{i}] {name:<20} weight={w:.4f}  (cap={WEIGHT_CAPS[i]:.1f})")

    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Combined CE + Dice Loss
# --------------------------------------------------------------------------- #
class CombinedLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, device: torch.device):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            ignore_index=IGNORE_INDEX,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        ce_loss   = self.ce(pred, target)
        dice_loss = self._dice_loss(pred, target)
        return 0.5 * ce_loss + 0.5 * dice_loss

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Soft dice loss averaged over all classes."""
        B, C, H, W = pred.shape
        prob = torch.softmax(pred, dim=1)          # [B, C, H, W]

        valid = (target != IGNORE_INDEX)            # [B, H, W]
        target_clamped = target.clone()
        target_clamped[~valid] = 0                  # safe range for one_hot

        target_oh = F.one_hot(target_clamped, num_classes=C)   # [B, H, W, C]
        target_oh = target_oh.permute(0, 3, 1, 2).float()      # [B, C, H, W]

        mask = valid.unsqueeze(1).float()           # [B, 1, H, W]
        prob      = prob * mask
        target_oh = target_oh * mask

        smooth = 1.0
        intersection = (prob * target_oh).sum(dim=(0, 2, 3))   # [C]
        sum_pred = prob.sum(dim=(0, 2, 3))
        sum_gt   = target_oh.sum(dim=(0, 2, 3))

        dice_per_class = 1.0 - (2.0 * intersection + smooth) / (sum_pred + sum_gt + smooth)
        return dice_per_class.mean()


# --------------------------------------------------------------------------- #
# Train / Val loops
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc='Train', ncols=90, leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, masks)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches  += 1
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp, num_classes):
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    metrics    = SegmentationMetrics(num_classes=num_classes, ignore_index=IGNORE_INDEX)

    for images, masks in tqdm(loader, desc='Val  ', ncols=90, leave=False):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, masks)

        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)
        total_loss += loss.item()
        n_batches  += 1

    results = metrics.compute()
    return total_loss / max(n_batches, 1), results


# --------------------------------------------------------------------------- #
# Prediction distribution (catch collapse early)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def pred_distribution(model, loader, device, num_classes):
    """Pixel-fraction predicted per class over full val set."""
    model.eval()
    counts = np.zeros(num_classes, dtype=np.int64)
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        preds  = model(images).argmax(dim=1).cpu().numpy().flatten()
        for c in range(num_classes):
            counts[c] += (preds == c).sum()
    total = counts.sum()
    if total == 0:
        return counts
    return counts / total


# --------------------------------------------------------------------------- #
# Comparison table
# --------------------------------------------------------------------------- #
def print_comparison(phase2_iou: list, phase2_miou: float, save_path: str):
    lines = [
        "Phase 1 vs Phase 2 Results",
        f"{'Class':<22} {'Phase1':>8}  {'Phase2':>8}  {'Change':>8}",
        "-" * 52,
    ]
    for i, name in enumerate(CLASS_NAMES):
        p1 = PHASE1_IOU[i] * 100
        p2 = phase2_iou[i] * 100
        ch = p2 - p1
        sign = "+" if ch >= 0 else ""
        lines.append(f"{name:<22} {p1:>7.2f}%  {p2:>7.2f}%  {sign}{ch:>6.2f}%")
    lines.append("-" * 52)
    p1m = PHASE1_MIOU * 100
    p2m = phase2_miou * 100
    ch  = p2m - p1m
    sign = "+" if ch >= 0 else ""
    lines.append(f"{'mIoU':<22} {p1m:>7.2f}%  {p2m:>7.2f}%  {sign}{ch:>6.2f}%")

    table = "\n".join(lines)
    print("\n" + table + "\n")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(table + "\n")
    print(f"[Done] Comparison saved to {save_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    set_seed(P2_CFG['seed'])

    # ── Paths ────────────────────────────────────────────────────────────────
    with open('config.yaml', 'r') as f:
        base_cfg = yaml.safe_load(f)

    train_images = base_cfg['data']['train_images']
    train_masks  = base_cfg['data']['train_masks']
    val_images   = base_cfg['data']['val_images']
    val_masks    = base_cfg['data']['val_masks']

    save_dir = Path(P2_CFG['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ──────────────────────────────────────────────────────────────
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = P2_CFG['mixed_precision'] and device.type == 'cuda'
    print(f"[Device]  {device}{'  (AMP enabled)' if use_amp else ''}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print(f"[GPU]     {torch.cuda.get_device_name(0)}")

    # ── Class weights ───────────────────────────────────────────────────────
    cw_cache = save_dir / 'class_weights_p2.pt'
    if cw_cache.exists():
        class_weights = torch.load(cw_cache, map_location='cpu', weights_only=True)
        print(f"[Weights] Loaded cached weights from {cw_cache}")
    else:
        class_weights = compute_class_weights_capped(train_masks)
        torch.save(class_weights, cw_cache)

    # ── Transforms ──────────────────────────────────────────────────────────
    h, w = P2_CFG['input_size']
    train_tfm = get_train_transform(h, w)
    val_tfm   = get_val_transform(h, w)

    # ── Datasets ────────────────────────────────────────────────────────────
    train_ds = OffRoadDataset(train_images, train_masks, transform=train_tfm)
    val_ds   = OffRoadDataset(val_images,   val_masks,   transform=val_tfm)
    print(f"[Data]    Train={len(train_ds)}  Val={len(val_ds)}")

    # ── WeightedRandomSampler ───────────────────────────────────────────────
    sample_weights = compute_sample_weights(
        train_masks, [str(f) for f in train_ds.image_files]
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    nw = P2_CFG['num_workers']
    pm = P2_CFG['pin_memory'] and device.type == 'cuda'
    train_loader = DataLoader(
        train_ds, batch_size=P2_CFG['batch_size'],
        sampler=sampler,
        num_workers=nw, pin_memory=pm, drop_last=True,
        persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=P2_CFG['batch_size'],
        shuffle=False,
        num_workers=nw, pin_memory=pm, drop_last=False,
        persistent_workers=(nw > 0),
    )

    # ── Model (fresh init, then load weights) ───────────────────────────────
    model = build_deeplabv3plus(
        backbone=P2_CFG['backbone'],
        encoder_weights=None,
        num_classes=P2_CFG['num_classes'],
    )
    model = model.to(device)

    # ── Loss, Optimizer, Scheduler ──────────────────────────────────────────
    criterion = CombinedLoss(class_weights, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=P2_CFG['learning_rate'],
        weight_decay=P2_CFG['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=P2_CFG['epochs'],
        eta_min=1e-7,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ── Logging ─────────────────────────────────────────────────────────────
    log_csv     = str(save_dir / 'training_log_p2.csv')
    csv_logger  = CSVLogger(log_csv)
    best_ckpt   = str(save_dir / 'best_model.pth')
    latest_ckpt = str(save_dir / 'latest_checkpoint.pth')

    best_miou   = 0.0
    best_iou    = [0.0] * P2_CFG['num_classes']
    no_improve  = 0
    start_epoch = 0
    patience    = int(P2_CFG['early_stopping_patience'])
    epochs      = int(P2_CFG['epochs'])
    num_classes = int(P2_CFG['num_classes'])

    # ── Resume from latest checkpoint (auto-detected) ────────────────────────
    if Path(latest_ckpt).exists():
        print(f"\n[Resume] Found {latest_ckpt} -- resuming Phase 2 training ...")
        resume = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        if use_amp and 'scaler' in resume:
            scaler.load_state_dict(resume['scaler'])
        start_epoch = int(resume['epoch']) + 1
        best_miou   = float(resume['best_miou'])
        best_iou    = list(resume.get('best_iou', [0.0] * num_classes))
        no_improve  = int(resume['no_improve'])
        print(f"[Resume] Continuing from epoch {start_epoch}  "
              f"(best mIoU so far={best_miou*100:.2f}%,  no_improve={no_improve})")
    else:
        # Fresh Phase 2 start — load Phase 1 model weights only
        p1_path = P2_CFG['checkpoint']
        if not Path(p1_path).exists():
            raise FileNotFoundError(
                f"Phase 1 checkpoint not found: {p1_path}\n"
                "Run Phase 1 first:  python train.py"
            )
        raw = torch.load(p1_path, map_location='cpu', weights_only=False)
        model.load_state_dict(raw['model'])
        p1_epoch = raw.get('epoch', '?')
        p1_miou  = raw.get('miou', 0.0)
        print(f"[Checkpoint] Loaded Phase 1 weights  "
              f"(epoch={p1_epoch}, mIoU={p1_miou*100:.2f}%)")

    print(f"\n{'='*65}")
    print(f"  Phase 2 fine-tuning: epochs {start_epoch}-{epochs-1}, "
          f"lr={P2_CFG['learning_rate']}")
    print(f"  To resume after stopping: just run the same command again.")
    print(f"{'='*65}\n")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # -- Train --
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )

        # -- Validate --
        val_loss, val_results = validate(
            model, val_loader, criterion, device, use_amp, num_classes
        )
        val_miou: float           = float(val_results['mIoU'])
        iou_classes: list[float]  = [float(x) for x in val_results['iou_per_class']]

        # -- Scheduler step --
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        elapsed = time.time() - t0

        # -- Prediction distribution check --
        dominant = max(range(num_classes), key=lambda c: iou_classes[c])
        dom_pct  = iou_classes[dominant] * 100
        if dom_pct > 90:
            print(f"  [!] COLLAPSE WARNING: [{dominant}]{CLASS_NAMES[dominant]} "
                  f"at {dom_pct:.1f}% IoU - other classes near zero")

        # -- Print epoch summary --
        print(f"E{epoch:03d}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"mIoU={val_miou*100:.2f}%  lr={current_lr:.2e}  t={elapsed:.0f}s")
        # Per-class IoU summary
        rare_str = "  ".join(
            f"{CLASS_NAMES[c][:4]}={iou_classes[c]*100:.1f}%"
            for c in [_LOGS, _ROCKS, _GROUND_CLUTTER, _DRY_BUSHES]
        )
        print(f"  Rare: {rare_str}")

        # -- Prediction distribution (every 5 epochs) --
        if epoch % 5 == 0:
            dist = pred_distribution(model, val_loader, device, num_classes)
            dist_str = "  ".join(
                f"{CLASS_NAMES[c][:4]}={dist[c]*100:.1f}%"
                for c in range(num_classes)
            )
            print(f"  PredDist: {dist_str}")

        # -- CSV logging --
        log_row = {
            'epoch':      epoch,
            'train_loss': round(train_loss, 5),
            'val_loss':   round(val_loss, 5),
            'val_miou':   round(val_miou, 5),
            'lr':         round(current_lr, 8),
            'time_s':     round(elapsed, 1),
        }
        for i, iou in enumerate(iou_classes):
            log_row[f'iou_cls{i}'] = round(iou, 5)
        csv_logger.log(log_row)

        # -- Checkpoint (best mIoU) --
        if val_miou > best_miou:
            best_miou = val_miou
            best_iou  = list(iou_classes)
            no_improve = 0
            ckpt = {
                'epoch':     epoch,
                'miou':      val_miou,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'iou_per_class': iou_classes,
            }
            torch.save(ckpt, best_ckpt)
            print(f"  [OK] New best mIoU={val_miou*100:.2f}% - saved to {best_ckpt}")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{patience}). Best={best_miou*100:.2f}%")

        # -- Early stopping --
        if no_improve >= patience:
            print(f"\n[Early Stop] No improvement for {patience} epochs.")
            break

    # --------------------------------------------------------------------------- #
    # Final comparison table
    # --------------------------------------------------------------------------- #
    print(f"\n{'='*65}")
    print(f"  Phase 2 complete.  Best val mIoU = {best_miou*100:.2f}%")
    print(f"{'='*65}")
    print_comparison(best_iou, best_miou, str(save_dir / 'phase_comparison.txt'))


if __name__ == '__main__':
    main()
