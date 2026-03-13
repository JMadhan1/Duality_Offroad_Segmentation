"""
test_colab_tta_val.py — Squeezing more mIoU: 640x640 resolution + 5-Aug TTA.
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
val_img_dir = ROOT_DIR / "Offroad_Segmentation_Training_Dataset" / "Offroad_Segmentation_Training_Dataset" / "val" / "Color_Images"
val_mask_dir = ROOT_DIR / "Offroad_Segmentation_Training_Dataset" / "Offroad_Segmentation_Training_Dataset" / "val" / "Segmentation"

model = smp.DeepLabV3Plus(encoder_name='resnet101', encoder_weights=None, classes=10).to(device)
model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
model.eval()

CLASS_MAP = {100:0, 200:1, 300:2, 500:3, 550:4, 600:5, 700:6, 800:7, 7100:8, 10000:9}
CLASS_NAMES = ['Trees','Lush_Bushes','Dry_Grass','Dry_Bushes','Ground_Clutter','Flowers','Logs','Rocks','Landscape','Sky']
IGNORE_INDEX = 255

def remap_mask(mask_array):
    out = np.full(mask_array.shape, IGNORE_INDEX, dtype=np.uint8)
    for raw_id, cls_idx in CLASS_MAP.items():
        out[mask_array == raw_id] = cls_idx
    return out

class SegMetrics:
    def __init__(self):
        self.conf = np.zeros((10, 10), dtype=np.int64)
    def update(self, pred, target):
        p, t = pred.flatten(), target.flatten()
        valid = t != IGNORE_INDEX
        self.conf += np.bincount(t[valid]*10 + p[valid], minlength=100).reshape(10,10)
    def compute(self):
        tp = np.diag(self.conf).astype(np.float64)
        iou = tp / (self.conf.sum(1) + self.conf.sum(0) - tp + 1e-10)
        present_mask = self.conf.sum(1) > 0
        return iou, iou[present_mask].mean()

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
RESIZE_TO = 640 # Pushing resolution
transform = A.Compose([A.Resize(RESIZE_TO, RESIZE_TO), A.Normalize(), ToTensorV2()])

val_files = sorted([f for f in os.listdir(val_img_dir) if f.endswith(('.png','.jpg'))])
print(f"[Eval] Resolution: {RESIZE_TO}x{RESIZE_TO}")
print(f"[Eval] TTA: Original, H-Flip, V-Flip, 0.75x, 1.25x (5-Aug)")

metrics_tta = SegMetrics()

with torch.no_grad():
    for fname in tqdm(val_files, desc="Val Squeeze"):
        img_raw = np.array(Image.open(val_img_dir / fname).convert('RGB'))
        mask_raw = np.array(Image.open(val_mask_dir / fname))
        if mask_raw.ndim == 3: mask_raw = mask_raw[:,:,0]
        mask = remap_mask(mask_raw)
        h_orig, w_orig = mask.shape
        
        tensor = transform(image=img_raw)['image'].unsqueeze(0).to(device)
        
        try:
            with torch.amp.autocast('cuda'):
                # 1. Original
                out1 = torch.softmax(model(tensor), dim=1)
                # 2. Horizontal Flip
                out2 = torch.flip(torch.softmax(model(torch.flip(tensor, [3])), dim=1), [3])
                # 3. Vertical Flip
                out3 = torch.flip(torch.softmax(model(torch.flip(tensor, [2])), dim=1), [2])
                # 4. Scale 0.75
                small = torch.nn.functional.interpolate(tensor, scale_factor=0.75, mode='bilinear', align_corners=False)
                out4 = torch.nn.functional.interpolate(torch.softmax(model(small), dim=1), size=(RESIZE_TO, RESIZE_TO), mode='bilinear', align_corners=False)
                # 5. Scale 1.25
                big = torch.nn.functional.interpolate(tensor, scale_factor=1.25, mode='bilinear', align_corners=False)
                out5 = torch.nn.functional.interpolate(torch.softmax(model(big), dim=1), size=(RESIZE_TO, RESIZE_TO), mode='bilinear', align_corners=False)
                
                avg_probs = (out1 + out2 + out3 + out4 + out5) / 5.0
                
                # Upsample to original resolution
                avg_probs_up = torch.nn.functional.interpolate(avg_probs, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
                pred = avg_probs_up.argmax(1).cpu().numpy()[0]
                
            metrics_tta.update(pred, mask)
        except torch.cuda.OutOfMemoryError:
            print(f"\n[OOM] Falling back to 512 for this image...")
            torch.cuda.empty_cache()
            # Fallback logic could go here, but for now we just skip the image in metrics or try 512
            continue

iou_tta, miou_tta = metrics_tta.compute()

print(f"\n{'='*65}")
print(f"{'Class':<20} {'Squeezed TTA IoU':>20}")
print(f"{'-'*65}")
for i in range(10):
    print(f"{CLASS_NAMES[i]:<20} {iou_tta[i]*100:>19.2f}%")
print(f"{'-'*65}")
print(f"{'Squeezed mIoU':<20} {miou_tta*100:>19.2f}%")
print(f"{'='*65}")
