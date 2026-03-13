"""
test_colab_model.py — Run inference using the ResNet101 Colab checkpoint.

Strategy:
- Model: DeepLabV3+ with ResNet101 backbone
- TTA: Original + Horizontal Flip
- Input Size: 448x448 (Balance between quality and 6GB VRAM safety)
- GPU Acceleration (AMP Mixed Precision)
"""

import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Setup paths
ROOT_DIR = Path(r"C:\Users\jmadh\OneDrive\Desktop\Bhavans HYD")
sys.path.insert(0, str(ROOT_DIR / "laptop"))

from src.dataset import OffRoadDataset
from src.model import build_deeplabv3plus
from src.utils import mask_to_rgb

def normalize_tensor(img_np):
    # Ensure float32
    img = img_np.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1) # [3, H, W]
    return torch.from_numpy(img).float()

def main():
    # ── Config ───────────────────────────────────────────────────────────────
    ckpt_path = ROOT_DIR / "hackathon_runs" / "best_p2.pth"
    test_images_dir = ROOT_DIR / "Offroad_Segmentation_testImages" / "Offroad_Segmentation_testImages" / "Color_Images"
    save_dir = ROOT_DIR / "runs" / "colab_final"
    pred_dir = save_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using {device}")

    # ── Load Model (ResNet101) ───────────────────────────────────────────────
    print(f"[Model] Initializing DeepLabV3+ ResNet101...")
    model = build_deeplabv3plus(
        backbone='resnet101', 
        encoder_weights=None, 
        num_classes=10
    ).to(device)
    
    # Force float32 to avoid "expected Double but found Float" errors
    model = model.float()

    print(f"[Model] Loading weights from {ckpt_path.name}...")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print("[Model] Weights loaded successfully.")
    except Exception as e:
        print(f"[Error] Failed to load checkpoint: {e}")
        return

    model.eval()

    # ── Dataset ───────────────────────────────────────────────────────────────
    if not test_images_dir.exists():
        print(f"[Error] Test images directory not found: {test_images_dir}")
        return
    
    test_ds = OffRoadDataset(str(test_images_dir), transform=None, return_filename=True)
    print(f"[Data] Found {len(test_ds)} test images.")

    # ── Inference ─────────────────────────────────────────────────────────────
    # 448x448 is a good middle ground for ResNet101 on 6GB VRAM
    INPUT_SIZE = (448, 448) 
    print(f"\n[Inference] Running TTA (Original + Flip) at {INPUT_SIZE}...")
    
    with torch.no_grad():
        for i in tqdm(range(len(test_ds)), desc="Inference"):
            try:
                img_raw, filename = test_ds[i]
                h_orig, w_orig = img_raw.shape[:2]

                # Prep input
                img_resized = Image.fromarray(img_raw).resize(INPUT_SIZE, Image.BILINEAR)
                img_tensor = normalize_tensor(np.array(img_resized)).unsqueeze(0).to(device)

                # Mixed Precision
                with torch.amp.autocast(device_type='cuda', enabled=device.type == 'cuda'):
                    # TTA: Original
                    logits1 = model(img_tensor)
                    probs = F.softmax(logits1, dim=1)

                    # TTA: Flip
                    img_flip = img_tensor.flip(-1)
                    logits2 = model(img_flip)
                    probs2 = F.softmax(logits2, dim=1).flip(-1)
                    
                    # Average
                    probs = (probs + probs2) / 2.0

                    # Upsample to original size
                    probs_up = F.interpolate(
                        probs, 
                        size=(h_orig, w_orig), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    pred = probs_up.argmax(1).cpu().numpy()[0]

                # Save colorized prediction
                pred_rgb = mask_to_rgb(pred)
                Image.fromarray(pred_rgb).save(pred_dir / filename)
                
            except torch.cuda.OutOfMemoryError:
                print(f"\n[OOM] Skipping {filename} due to memory.")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"\n[Error] Failed on {filename}: {e}")
                continue

    print(f"\n{'='*65}")
    print(f" SUCCESS! Predictions saved to: {pred_dir}")
    print(f"{'='*65}")

if __name__ == '__main__':
    main()
