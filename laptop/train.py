"""
train.py — Full training pipeline for Offroad Segmentation.

Usage:
    python train.py                          # uses config.yaml
    python train.py --config config.yaml
    python train.py --arch segformer         # override architecture
    python train.py --resume runs/best_model.pth

Features:
  - Mixed-precision (AMP) training
  - Gradient accumulation
  - Linear LR warmup → CosineAnnealingWarmRestarts
  - EMA of model weights
  - Best model checkpointing (val mIoU)
  - Per-epoch CSV + TensorBoard logging
  - Per-class IoU printed every epoch
  - Early stopping
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# ── local imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.utils      import load_config, set_seed, EMA, save_checkpoint, CSVLogger, plot_training_curves
from src.dataset    import OffRoadDataset, compute_class_weights, build_dataloaders, CLASS_NAMES, NUM_CLASSES, IGNORE_INDEX
from src.transforms import get_train_transforms, get_val_transforms
from src.losses     import build_criterion
from src.metrics    import SegmentationMetrics, print_metrics
from src.model      import build_model


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_optimizer(model: nn.Module, cfg: dict):
    tr  = cfg['training']
    opt = tr.get('optimizer', 'adamw').lower()
    lr  = tr['learning_rate']
    wd  = tr.get('weight_decay', 0.01)
    if opt == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    raise ValueError(f"Unknown optimizer: {opt}")


def get_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    tr    = cfg['training']
    sched = tr.get('scheduler', 'reduce_on_plateau').lower()
    epochs = tr['epochs']

    if sched == 'reduce_on_plateau':
        # Reactive scheduler: halves LR when val mIoU stops improving.
        # Caller must call scheduler.step(val_miou) after each validation.
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',                            # maximise val mIoU
            factor=tr.get('reduce_factor', 0.5),
            patience=tr.get('reduce_patience', 5),
            min_lr=tr.get('reduce_min_lr', 1e-6),
        ), 'plateau'
    elif sched == 'cosine_warmrestarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=tr.get('T_0', 10),
            T_mult=tr.get('T_mult', 2),
            eta_min=tr.get('eta_min', 1e-7),
        ), 'epoch'
    elif sched == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=tr['learning_rate'],
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=0.1,
        ), 'step'
    elif sched == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=tr.get('eta_min', 1e-7)
        ), 'epoch'
    raise ValueError(f"Unknown scheduler: {sched}")


# ─────────────────────────────────────────────────────────────────────────────
# Train one epoch
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(
    model, loader, criterion, optimizer, scaler,
    device, cfg, scheduler, sched_mode, epoch,
    ema=None,
):
    model.train()
    tr         = cfg['training']
    accum      = tr.get('grad_accum_steps', 1)
    clip_norm  = tr.get('gradient_clip', 1.0)
    use_amp    = tr.get('mixed_precision', True) and device.type == 'cuda'
    amp_dtype  = torch.float16 if use_amp else torch.float32
    log_every  = cfg['output'].get('log_interval', 20)
    num_classes = cfg['classes']['num_classes']

    metrics = SegmentationMetrics(num_classes, IGNORE_INDEX)
    total_loss = 0.0
    loss_components = {'ce': 0., 'dice': 0., 'focal': 0.}
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Train E{epoch}", leave=False, dynamic_ncols=True)
    for step, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            logits = model(images)
            loss, comps = criterion(logits, masks)
            loss = loss / accum

        scaler.scale(loss).backward()

        if (step + 1) % accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema is not None:
                ema.update()
            if sched_mode == 'step':
                scheduler.step()

        # Accumulate metrics
        preds = logits.argmax(dim=1).detach()
        metrics.update(preds, masks)
        total_loss += loss.item() * accum
        for k in loss_components:
            loss_components[k] += comps[k]

        if (step + 1) % log_every == 0:
            pbar.set_postfix({'loss': f"{comps['total']:.4f}"})

    n = len(loader)
    results = metrics.compute()
    return {
        'loss':        total_loss / n,
        'ce_loss':     loss_components['ce'] / n,
        'dice_loss':   loss_components['dice'] / n,
        'focal_loss':  loss_components['focal'] / n,
        'miou':        results['mIoU'],
        'iou_per_class': results['iou_per_class'],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    model.eval()
    use_amp     = cfg['training'].get('mixed_precision', True) and device.type == 'cuda'
    amp_dtype   = torch.float16 if use_amp else torch.float32
    num_classes = cfg['classes']['num_classes']

    metrics    = SegmentationMetrics(num_classes, IGNORE_INDEX)
    total_loss = 0.0
    loss_components = {'ce': 0., 'dice': 0., 'focal': 0.}

    for images, masks in tqdm(loader, desc="Val  ", leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            logits = model(images)
            loss, comps = criterion(logits, masks)

        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)
        total_loss += loss.item()
        for k in loss_components:
            loss_components[k] += comps[k]

    n = len(loader)
    results = metrics.compute()
    return {
        'loss':        total_loss / n,
        'ce_loss':     loss_components['ce'] / n,
        'dice_loss':   loss_components['dice'] / n,
        'focal_loss':  loss_components['focal'] / n,
        'miou':        results['mIoU'],
        'iou_per_class': results['iou_per_class'],
        'full_results':  results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Train Offroad Segmentation')
    parser.add_argument('--config',  default='config.yaml', help='Path to config YAML')
    parser.add_argument('--arch',    default=None,  help='Override model.architecture')
    parser.add_argument('--epochs',  default=None,  type=int, help='Override epochs')
    parser.add_argument('--resume',  default=None,  help='Checkpoint to resume from')
    args = parser.parse_args()

    # ── Config ──────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    if args.arch:
        cfg['model']['architecture'] = args.arch
    if args.epochs:
        cfg['training']['epochs'] = args.epochs

    set_seed(cfg['training'].get('seed', 42))

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")
    if device.type == 'cuda':
        print(f"[GPU]    {torch.cuda.get_device_name(0)}")
        # Enable cudnn auto-tuner for fixed input sizes → significant speedup
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    # ── Output dirs ─────────────────────────────────────────────────────────
    runs_dir = Path(cfg['output']['runs_dir'])
    runs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path     = runs_dir / cfg['output']['checkpoint_name']
    ema_ckpt_path = runs_dir / cfg['output']['ema_checkpoint_name']
    log_csv       = cfg['output']['log_csv']

    # ── Transforms & Dataloaders ─────────────────────────────────────────────
    input_size = tuple(cfg['training']['input_size'])
    train_tfm  = get_train_transforms(input_size)
    val_tfm    = get_val_transforms(input_size)
    train_loader, val_loader = build_dataloaders(cfg, train_tfm, val_tfm)
    print(f"[Data]   Train={len(train_loader.dataset)}  Val={len(val_loader.dataset)}")

    # ── Class weights ────────────────────────────────────────────────────────
    class_weights = None
    if cfg['loss'].get('use_class_weights', True):
        weights_file = runs_dir / 'class_weights.pt'
        if weights_file.exists():
            class_weights = torch.load(weights_file)
            print(f"[Weights] Loaded cached class weights from {weights_file}")
        else:
            class_weights = compute_class_weights(cfg['data']['train_masks'])
            torch.save(class_weights, weights_file)
        class_weights = class_weights.to(device)

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)

    # ── Compile model for faster execution (PyTorch 2.x) ───────────────────
    # Note: torch.compile disabled on Windows (requires Triton / long compile times)

    # ── Loss, Optimizer, Scheduler ──────────────────────────────────────────
    criterion = build_criterion(cfg, class_weights).to(device)
    optimizer = get_optimizer(model, cfg)
    scaler    = GradScaler('cuda', enabled=cfg['training'].get('mixed_precision', True) and device.type == 'cuda')
    scheduler, sched_mode = get_scheduler(optimizer, cfg, len(train_loader))

    # ── EMA ─────────────────────────────────────────────────────────────────
    ema_decay = cfg['training'].get('ema_decay', 0.9999)
    ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None

    # ── TensorBoard ─────────────────────────────────────────────────────────
    writer = None
    if cfg['output'].get('tensorboard', True):
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=str(runs_dir / 'tb_logs'))
            print(f"[TB]     tensorboard --logdir {runs_dir/'tb_logs'}")
        except ImportError:
            print("[TB]     tensorboard not available, skipping")

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    best_miou   = 0.0
    if args.resume and Path(args.resume).exists():
        from src.utils import load_checkpoint
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler, ema, str(device))
        start_epoch = ckpt.get('epoch', 0) + 1
        best_miou   = ckpt.get('miou', 0.0)

    # ── Training loop ────────────────────────────────────────────────────────
    csv_logger   = CSVLogger(log_csv)
    patience   = cfg['training'].get('early_stopping_patience', 20)
    no_improve = 0
    epochs     = cfg['training']['epochs']
    num_classes = cfg['classes']['num_classes']

    print(f"\n{'='*65}")
    print(f"  Starting training: {epochs} epochs, input {input_size}")
    print(f"{'='*65}\n")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # Train
        train_results = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, cfg, scheduler, sched_mode, epoch, ema,
        )

        # Step epoch-level schedulers (not plateau — that needs val_miou)
        if sched_mode == 'epoch':
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # ── Validate with EMA weights if available ──────────────────────────
        if ema is not None:
            ema.apply_shadow()
        val_results = validate(model, val_loader, criterion, device, cfg)
        if ema is not None:
            ema.restore()

        val_miou = val_results['miou']
        elapsed  = time.time() - t0

        # ── ReduceLROnPlateau needs the metric ───────────────────────────────
        if sched_mode == 'plateau':
            scheduler.step(val_miou)
            current_lr = optimizer.param_groups[0]['lr']   # may have changed

        # ── Prediction distribution check — catch collapse early ─────────────
        pred_dist = val_results['full_results']['iou_per_class']
        dominant  = max(range(num_classes), key=lambda c: pred_dist[c])
        dom_pct   = pred_dist[dominant] * 100
        if dom_pct > 60:
            print(f"  [!] COLLAPSE WARNING: [{dominant}]{CLASS_NAMES[dominant]} "
                  f"dominates val at {dom_pct:.1f}% IoU - other classes near zero")

        # ── Logging ─────────────────────────────────────────────────────────
        log_row = {
            'epoch':       epoch,
            'train_loss':  round(train_results['loss'], 5),
            'val_loss':    round(val_results['loss'], 5),
            'train_miou':  round(train_results['miou'], 5),
            'val_miou':    round(val_miou, 5),
            'lr':          round(current_lr, 8),
            'time_s':      round(elapsed, 1),
        }
        for i, iou in enumerate(val_results['iou_per_class']):
            log_row[f'iou_cls{i}'] = round(iou, 5)
        csv_logger.log(log_row)

        if writer:
            writer.add_scalar('Loss/train', train_results['loss'], epoch)
            writer.add_scalar('Loss/val',   val_results['loss'],   epoch)
            writer.add_scalar('mIoU/train', train_results['miou'], epoch)
            writer.add_scalar('mIoU/val',   val_miou,              epoch)
            writer.add_scalar('LR',         current_lr,            epoch)

        print(f"\nEpoch {epoch:3d}/{epochs}  "
              f"Train: loss={train_results['loss']:.4f} mIoU={train_results['miou']*100:.1f}%  "
              f"Val: loss={val_results['loss']:.4f} mIoU={val_miou*100:.2f}%  "
              f"LR={current_lr:.2e}  [{elapsed:.0f}s]")

        print_metrics(val_results['full_results'], CLASS_NAMES, epoch=epoch)

        # ── Checkpointing ────────────────────────────────────────────────────
        if val_miou > best_miou:
            best_miou  = val_miou
            no_improve = 0
            save_checkpoint(str(ckpt_path), model, optimizer, scheduler, epoch, val_miou, ema)
            if ema is not None:
                # Save EMA-applied weights separately
                ema.apply_shadow()
                save_checkpoint(str(ema_ckpt_path), model, optimizer, scheduler, epoch, val_miou)
                ema.restore()
            print(f"  [OK] New best mIoU={float(best_miou)*100:.2f}% - checkpoint saved")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{patience}). Best={float(best_miou)*100:.2f}%")

        # ── Early stopping ────────────────────────────────────────────────────
        if no_improve >= patience:
            print(f"\n[Early Stop] No improvement for {patience} epochs. Stopping.")
            break

    # ── Final ────────────────────────────────────────────────────────────────
    if writer:
        writer.close()

    print(f"\n{'='*65}")
    print(f"  Training complete!  Best val mIoU = {float(best_miou)*100:.2f}%")
    print(f"  Best checkpoint -> {ckpt_path}")
    if ema is not None:
        print(f"  EMA checkpoint  -> {ema_ckpt_path}")
    print(f"{'='*65}")

    # Plot training curves
    if Path(log_csv).exists():
        plot_training_curves(log_csv, str(runs_dir))


if __name__ == '__main__':
    main()
