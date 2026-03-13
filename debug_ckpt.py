import torch
from pathlib import Path
import sys

ROOT_DIR = Path(r"C:\Users\jmadh\OneDrive\Desktop\Bhavans HYD")
sys.path.insert(0, str(ROOT_DIR / "laptop"))
from src.model import build_deeplabv3plus

ckpt_path = ROOT_DIR / "hackathon_runs" / "best_p2.pth"
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

print("Checkpoint keys:", checkpoint.keys() if isinstance(checkpoint, dict) else "Not a dict")
if isinstance(checkpoint, dict) and 'model' in checkpoint:
    state_dict = checkpoint['model']
    print("Found 'model' key in checkpoint.")
else:
    state_dict = checkpoint

# Try to load into model
model = build_deeplabv3plus(backbone='resnet101', num_classes=10)
try:
    model.load_state_dict(state_dict)
    print("Success! State dict loaded into ResNet101 DeepLabV3+.")
except Exception as e:
    print("Error loading state dict:", str(e)[:300], "...")

# Check if maybe it's ResNet34
model34 = build_deeplabv3plus(backbone='resnet34', num_classes=10)
try:
    model34.load_state_dict(state_dict)
    print("Success! State dict loaded into ResNet34 DeepLabV3+.")
except Exception as e:
    # print("Error loading into ResNet34:", str(e)[:100])
    pass
