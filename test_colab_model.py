"""
test_colab_model.py — Final Squeezed Submission (640x640 + 5-Aug TTA).
Output: Raw uint16 masks for leaderboard.
"""

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import segmentation_models_pytorch as smp
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_DIR = Path(r"C:\Users\jmadh\OneDrive\Desktop\Bhavans HYD")

ckpt_path = ROOT_DIR / "hackathon_runs" / "best_p2.pth"
test_img_dir = ROOT_DIR / "Offroad_Segmentation_testImages" / "Offroad_Segmentation_testImages" / "Color_Images"
pred_dir = ROOT_DIR / "runs/colab_final/predictions"
os.makedirs(pred_dir, exist_ok=True)

model = smp.DeepLabV3Plus(encoder_name='resnet101', encoder_weights=None, classes=10).to(device)
model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
model.eval()

# Leaderboard Label Mapping
CLASS_MAP = {100:0, 200:1, 300:2, 500:3, 550:4, 600:5, 700:6, 800:7, 7100:8, 10000:9}
INV_MAP = {v:k for k,v in CLASS_MAP.items()}

# Squeezed resolution
RESIZE_TO = 640
transform = A.Compose([A.Resize(RESIZE_TO, RESIZE_TO), A.Normalize(), ToTensorV2()])

test_files = sorted([f for f in os.listdir(test_img_dir) if f.endswith(('.png','.jpg'))])
print(f"[Inference] Squeezing mIoU with {RESIZE_TO}x{RESIZE_TO} resolution...")
print(f"[Inference] TTA: 5-Augmentations (Orig, H-Flip, V-Flip, 0.75x, 1.25x)")

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE LOOP
# ─────────────────────────────────────────────────────────────────────────────
with torch.no_grad():
    for fname in tqdm(test_files, desc="Final Submission"):
        img = np.array(Image.open(test_img_dir / fname).convert('RGB'))
        h_orig, w_orig = img.shape[:2]
        
        tensor = transform(image=img)['image'].unsqueeze(0).to(device)
        
        with torch.amp.autocast('cuda'):
            # TTA Ensemble
            out1 = torch.softmax(model(tensor), dim=1)
            out2 = torch.flip(torch.softmax(model(torch.flip(tensor, [3])), dim=1), [3])   # H-Flip
            out3 = torch.flip(torch.softmax(model(torch.flip(tensor, [2])), dim=1), [2])   # V-Flip
            
            small = torch.nn.functional.interpolate(tensor, scale_factor=0.75, mode='bilinear', align_corners=False)
            out4 = torch.nn.functional.interpolate(torch.softmax(model(small), dim=1), size=(RESIZE_TO, RESIZE_TO), mode='bilinear', align_corners=False)
            
            big = torch.nn.functional.interpolate(tensor, scale_factor=1.25, mode='bilinear', align_corners=False)
            out5 = torch.nn.functional.interpolate(torch.softmax(model(big), dim=1), size=(RESIZE_TO, RESIZE_TO), mode='bilinear', align_corners=False)
            
            avg_probs = (out1 + out2 + out3 + out4 + out5) / 5.0
            
            # Upsample back to original resolution
            avg_probs_up = torch.nn.functional.interpolate(avg_probs, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
            pred = avg_probs_up.argmax(1).cpu().numpy()[0]
        
        # Map back to raw labels
        final = np.zeros_like(pred, dtype=np.uint16)
        for c, r in INV_MAP.items():
            final[pred == c] = r
        
        # Save raw mask
        Image.fromarray(final).save(pred_dir / fname)

print(f"\nDONE! Final {len(test_files)} predictions saved to {pred_dir}")
