"""
test_squeeze_more.py — Validating 8-Aug Multi-Scale TTA to squeeze final mIoU.
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
RESIZE_TO = 640
transform = A.Compose([A.Resize(RESIZE_TO, RESIZE_TO), A.Normalize(), ToTensorV2()])

val_files = sorted([f for f in os.listdir(val_img_dir) if f.endswith(('.png','.jpg'))])
print(f"[Eval] Resolution: {RESIZE_TO}x{RESIZE_TO}")
print(f"[Eval] Ultra-TTA: 8-Augmentations (0.5x, 0.75x, 1.0x, 1.25x, 1.5x, H-Flip, V-Flip, H-Flip+1.25x)")

metrics_tta = SegMetrics()

with torch.no_grad():
    for fname in tqdm(val_files, desc="Ultra Squeeze"):
        img_raw = np.array(Image.open(val_img_dir / fname).convert('RGB'))
        mask_raw = np.array(Image.open(val_mask_dir / fname))
        if mask_raw.ndim == 3: mask_raw = mask_raw[:,:,0]
        mask = remap_mask(mask_raw)
        h_orig, w_orig = mask.shape
        
        tensor = transform(image=img_raw)['image'].unsqueeze(0).to(device)
        
        try:
            with torch.amp.autocast('cuda'):
                # 1. Original (1.0x)
                out1 = torch.softmax(model(tensor), dim=1)
                
                # 2. Horizontal Flip
                out2 = torch.flip(torch.softmax(model(torch.flip(tensor, [3])), dim=1), [3])
                
                # 3. Vertical Flip
                out3 = torch.flip(torch.softmax(model(torch.flip(tensor, [2])), dim=1), [2])
                
                # 4. Scale 0.75x
                s75 = torch.nn.functional.interpolate(tensor, scale_factor=0.75, mode='bilinear', align_corners=False)
                out4 = torch.nn.functional.interpolate(torch.softmax(model(s75), dim=1), size=(RESIZE_TO, RESIZE_TO), mode='bilinear', align_corners=False)
                
                # 5. Scale 1.25x
                s125 = torch.nn.functional.interpolate(tensor, scale_factor=1.25, mode='bilinear', align_corners=False)
                out5 = torch.nn.functional.interpolate(torch.softmax(model(s125), dim=1), size=(RESIZE_TO, RESIZE_TO), mode='bilinear', align_corners=False)
                
                # 6. Scale 0.5x
                s50 = torch.nn.functional.interpolate(tensor, scale_factor=0.5, mode='bilinear', align_corners=False)
                out6 = torch.nn.functional.interpolate(torch.softmax(model(s50), dim=1), size=(RESIZE_TO, RESIZE_TO), mode='bilinear', align_corners=False)
                
                # 7. Scale 1.5x (Check OOM)
                s150 = torch.nn.functional.interpolate(tensor, scale_factor=1.5, mode='bilinear', align_corners=False)
                out7 = torch.nn.functional.interpolate(torch.softmax(model(s150), dim=1), size=(RESIZE_TO, RESIZE_TO), mode='bilinear', align_corners=False)
                
                # 8. H-Flip + Scale 1.25x
                tensor_hf = torch.flip(tensor, [3])
                s125hf = torch.nn.functional.interpolate(tensor_hf, scale_factor=1.25, mode='bilinear', align_corners=False)
                out8 = torch.nn.functional.interpolate(torch.flip(torch.softmax(model(s125hf), dim=1), [3]), size=(RESIZE_TO, RESIZE_TO), mode='bilinear', align_corners=False)
                
                avg_probs = (out1 + out2 + out3 + out4 + out5 + out6 + out7 + out8) / 8.0
                
                # Upsample to original resolution
                avg_probs_up = torch.nn.functional.interpolate(avg_probs, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
                pred = avg_probs_up.argmax(1).cpu().numpy()[0]
                
            metrics_tta.update(pred, mask)
            
        except torch.cuda.OutOfMemoryError:
            print(f"\n[OOM] Scale 1.5x failed on {fname}. Falling back to 7-Aug.")
            torch.cuda.empty_cache()
            with torch.amp.autocast('cuda'):
                avg_probs = (out1 + out2 + out3 + out4 + out5 + out6 + out8) / 7.0
                avg_probs_up = torch.nn.functional.interpolate(avg_probs, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
                pred = avg_probs_up.argmax(1).cpu().numpy()[0]
            metrics_tta.update(pred, mask)

iou_tta, miou_tta = metrics_tta.compute()

print(f"\n{'='*65}")
print(f"{'Class':<20} {'Ultra-TTA IoU':>20}")
print(f"{'-'*65}")
for i in range(10):
    print(f"{CLASS_NAMES[i]:<20} {iou_tta[i]*100:>19.2f}%")
print(f"{'-'*65}")
print(f"{'Ultra mIoU':<20} {miou_tta*100:>19.2f}%")
print(f"{'='*65}")
print(f"Comparison: Previous 63.89% vs Current {miou_tta*100:.2f}%")
