import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

try:
    h, w = 512, 512
    tfm = A.Compose([
        A.RandomResizedCrop(
            size=(h, w),
            scale=(0.4, 1.0),
            p=1.0,
        ),
        ToTensorV2(),
    ])
    print("Successfully created transform")
except Exception as e:
    print(f"Error: {e}")
