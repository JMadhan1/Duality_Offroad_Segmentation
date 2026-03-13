"""
test.py — Inference on test images with Test-Time Augmentation (TTA).

Usage:
    python test.py                                         # uses config.yaml defaults
    python test.py --checkpoint runs/best_model_ema.pth
    python test.py --input ../Offroad_Segmentation_testImages/.../Color_Images
    python test.py --no-tta                                # disable TTA (faster)

Outputs (in --output dir):
    masks/          — raw class-index PNGs  (values 0-9)
    masks_color/    — colourised class PNGs
    overlays/       — semi-transparent overlay on original RGB
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from src.utils      import load_config, mask_to_rgb, save_overlay, CLASS_COLORS
from src.dataset    import OffRoadDataset, IGNORE_INDEX, NUM_CLASSES
from src.transforms import get_test_transforms
from src.model      import build_model, multiscale_predict


# ─────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, cfg: dict, device: torch.device):
    model = build_model(cfg).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    epoch = ckpt.get('epoch', '?')
    miou  = ckpt.get('miou', 0.0)
    print(f"[Checkpoint] Loaded {checkpoint_path}  epoch={epoch}  val_mIoU={miou*100:.2f}%")
    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model,
    images_dir: str,
    output_dir: str,
    cfg: dict,
    device: torch.device,
    use_tta: bool = True,
):
    inf_cfg    = cfg.get('inference', {})
    input_size = tuple(cfg['training']['input_size'])
    tta_scales = inf_cfg.get('tta_scales', [0.75, 1.0, 1.25])
    tta_flip   = inf_cfg.get('tta_flips', True)
    use_amp    = cfg['training'].get('mixed_precision', True) and device.type == 'cuda'

    out = Path(output_dir)
    masks_dir        = out / 'masks'
    masks_color_dir  = out / 'masks_color'
    overlays_dir     = out / 'overlays'
    for d in [masks_dir, masks_color_dir, overlays_dir]:
        d.mkdir(parents=True, exist_ok=True)

    tfm = get_test_transforms(input_size)
    ds  = OffRoadDataset(images_dir=images_dir, transform=tfm, return_filename=True)
    print(f"[Inference] {len(ds)} images from {images_dir}")
    print(f"[TTA]       {'enabled (scales=' + str(tta_scales) + ')' if use_tta else 'disabled'}")

    import time
    total_time = 0.0

    for idx in tqdm(range(len(ds)), desc='Inference', dynamic_ncols=True):
        img_tensor, filename = ds[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        t0 = time.time()
        if use_tta:
            # multiscale_predict handles multi-scale + horizontal flip averaging
            with torch.cuda.amp.autocast(enabled=use_amp):
                probs = multiscale_predict(model, img_tensor, scales=tta_scales, flip=tta_flip)
        else:
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)

        elapsed = time.time() - t0
        total_time += elapsed

        pred_mask = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # [H, W]

        # Load original RGB for overlay (at original resolution)
        orig_img = np.array(Image.open(Path(images_dir) / filename).convert('RGB'))

        # Resize pred_mask back to original image resolution if needed
        if pred_mask.shape != orig_img.shape[:2]:
            pred_pil = Image.fromarray(pred_mask).resize(
                (orig_img.shape[1], orig_img.shape[0]), Image.NEAREST
            )
            pred_mask_orig = np.array(pred_pil)
        else:
            pred_mask_orig = pred_mask

        stem = Path(filename).stem

        # Save raw class-index mask
        Image.fromarray(pred_mask_orig).save(masks_dir / f"{stem}.png")

        # Save colourised mask
        color_mask = mask_to_rgb(pred_mask_orig)
        Image.fromarray(color_mask).save(masks_color_dir / f"{stem}.png")

        # Save overlay
        save_overlay(orig_img, pred_mask_orig, str(overlays_dir / f"{stem}.png"), alpha=0.45)

    avg_ms = (total_time / len(ds)) * 1000
    print(f"\n[Speed]  Avg inference: {avg_ms:.1f} ms/image")
    print(f"[Output] Saved to {output_dir}/")
    print(f"  masks/        — raw class-index PNGs")
    print(f"  masks_color/  — colourised PNGs")
    print(f"  overlays/     — semi-transparent overlays")
    return avg_ms


def main():
    parser = argparse.ArgumentParser(description='Inference on test images')
    parser.add_argument('--config',     default='config.yaml')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to .pth checkpoint (default: best_model_ema.pth or best_model.pth)')
    parser.add_argument('--input',      default=None,
                        help='Override test images directory from config')
    parser.add_argument('--output',     default=None,
                        help='Override output directory from config')
    parser.add_argument('--no-tta',     action='store_true', help='Disable TTA')
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # Checkpoint selection
    runs_dir = Path(cfg['output']['runs_dir'])
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ema_path = runs_dir / cfg['output'].get('ema_checkpoint_name', 'best_model_ema.pth')
        reg_path = runs_dir / cfg['output']['checkpoint_name']
        ckpt_path = str(ema_path) if ema_path.exists() else str(reg_path)
        print(f"[Auto]   Using checkpoint: {ckpt_path}")

    model = load_model(ckpt_path, cfg, device)

    images_dir = args.input  or cfg['data']['test_images']
    output_dir = args.output or cfg['inference'].get('output_dir', 'runs/predictions')
    use_tta    = not args.no_tta

    run_inference(model, images_dir, output_dir, cfg, device, use_tta=use_tta)


if __name__ == '__main__':
    main()
