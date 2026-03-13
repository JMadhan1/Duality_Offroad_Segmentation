"""
final_test_predictions.py — Final TTA predictions: 1002 images @ 640x640, 5-Aug TTA.
Generated for submission leaderboard (raw uint16 masks).
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
device = torch.device('cuda')
ROOT_DIR = Path(r"C:\Users\jmadh\OneDrive\Desktop\Bhavans HYD")

# Load Colab's 63%+ model
ckpt_path = ROOT_DIR / "hackathon_runs" / "best_p2.pth"
model = smp.DeepLabV3Plus(encoder_name='resnet101', encoder_weights=None, classes=10).to(device)
model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
model.eval()

CLASS_MAP = {100:0, 200:1, 300:2, 500:3, 550:4, 600:5, 700:6, 800:7, 7100:8, 10000:9}
INV_MAP = {v:k for k,v in CLASS_MAP.items()}
transform = A.Compose([A.Resize(640, 640), A.Normalize(), ToTensorV2()])

TEST_IMG = ROOT_DIR / "Offroad_Segmentation_testImages" / "Offroad_Segmentation_testImages" / "Color_Images"
pred_dir = ROOT_DIR / 'runs' / 'final_submission' / 'predictions'
os.makedirs(pred_dir, exist_ok=True)

test_files = sorted([f for f in os.listdir(TEST_IMG) if f.endswith(('.png','.jpg'))])
print(f"Final TTA predictions: {len(test_files)} images @ 640x640, 5-Aug TTA")

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
for fname in tqdm(test_files):
    img = np.array(Image.open(f"{TEST_IMG}/{fname}").convert('RGB'))
    h_orig, w_orig = img.shape[:2]
    tensor = transform(image=img)['image'].unsqueeze(0).to(device)
    
    with torch.no_grad(), torch.amp.autocast('cuda'):
        # Original
        out1 = torch.softmax(model(tensor), dim=1)
        # H-Flip
        out2 = torch.flip(torch.softmax(model(torch.flip(tensor, [3])), dim=1), [3])
        # V-Flip
        out3 = torch.flip(torch.softmax(model(torch.flip(tensor, [2])), dim=1), [2])
        # 0.75x Scale
        small = torch.nn.functional.interpolate(tensor, scale_factor=0.75, mode='bilinear', align_corners=False)
        out4 = torch.nn.functional.interpolate(torch.softmax(model(small), dim=1), size=(640, 640), mode='bilinear', align_corners=False)
        # 1.25x Scale
        big = torch.nn.functional.interpolate(tensor, scale_factor=1.25, mode='bilinear', align_corners=False)
        out5 = torch.nn.functional.interpolate(torch.softmax(model(big), dim=1), size=(640, 640), mode='bilinear', align_corners=False)
        
        # Average & Upsample
        avg_probs = (out1 + out2 + out3 + out4 + out5) / 5.0
        avg_probs_up = torch.nn.functional.interpolate(avg_probs, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
        pred = avg_probs_up.argmax(1).cpu().numpy()[0]
    
    # Map to raw uint16 values
    final = np.zeros_like(pred, dtype=np.uint16)
    for c, r in INV_MAP.items(): 
        final[pred == c] = r
    
    Image.fromarray(final).save(f"{pred_dir}/{fname}")

print(f"\n✅ DONE! {len(os.listdir(pred_dir))} predictions → {pred_dir}")
