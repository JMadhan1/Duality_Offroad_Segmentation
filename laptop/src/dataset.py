"""
dataset.py — OffRoad Segmentation Dataset

Handles loading, class-ID remapping, and DataLoader construction
for the Duality AI Offroad Segmentation dataset.

Raw mask pixel values → contiguous class indices 0-9:
  100  → 0  Trees
  200  → 1  Lush Bushes
  300  → 2  Dry Grass
  500  → 3  Dry Bushes
  550  → 4  Ground Clutter
  600  → 5  Flowers
  700  → 6  Logs
  800  → 7  Rocks
  7100 → 8  Landscape
  10000→ 9  Sky
  any other value → 255 (ignore_index)
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------------------------------- #
# Class definitions
# --------------------------------------------------------------------------- #
CLASS_MAP = {
    100:   0,   # Trees
    200:   1,   # Lush Bushes
    300:   2,   # Dry Grass
    500:   3,   # Dry Bushes
    550:   4,   # Ground Clutter
    600:   5,   # Flowers
    700:   6,   # Logs
    800:   7,   # Rocks
    7100:  8,   # Landscape
    10000: 9,   # Sky
}

# Inverse map: class index → raw pixel ID  (used for saving predictions)
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

CLASS_NAMES = [
    'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky',
]

NUM_CLASSES  = 10
IGNORE_INDEX = 255

# --------------------------------------------------------------------------- #
# Fast look-up table for mask remapping
# --------------------------------------------------------------------------- #
_LUT_SIZE = 10001   # max raw class ID (10000) + 1
_REMAP_LUT = np.full(_LUT_SIZE, IGNORE_INDEX, dtype=np.uint8)
for _raw_id, _cls_idx in CLASS_MAP.items():
    _REMAP_LUT[_raw_id] = _cls_idx


def remap_mask(mask_array: np.ndarray) -> np.ndarray:
    """Remap raw pixel values → contiguous class indices using a LUT.

    Any value not in CLASS_MAP (including 0) is mapped to IGNORE_INDEX (255).
    Handles masks stored as uint8, uint16, int32, etc.
    """
    mask_int = mask_array.astype(np.int32)
    # Values >= _LUT_SIZE are very unlikely; map them to IGNORE_INDEX directly
    valid = (mask_int >= 0) & (mask_int < _LUT_SIZE)
    out = np.full(mask_array.shape, IGNORE_INDEX, dtype=np.uint8)
    out[valid] = _REMAP_LUT[mask_int[valid]]
    return out


def restore_mask(class_mask: np.ndarray) -> np.ndarray:
    """Remap contiguous class indices → original raw pixel IDs (for submission)."""
    out = np.zeros_like(class_mask, dtype=np.int32)
    for cls_idx, raw_id in INV_CLASS_MAP.items():
        out[class_mask == cls_idx] = raw_id
    return out


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class OffRoadDataset(Dataset):
    """PyTorch Dataset for Offroad Segmentation.

    Args:
        images_dir: Path to folder containing RGB .png images.
        masks_dir:  Path to folder containing segmentation mask .png files.
                    Pass None for test mode (no ground-truth masks).
        transform:  Albumentations Compose transform (handles image + mask).
        return_filename: If True, __getitem__ also returns the image filename.
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str | None = None,
        transform=None,
        return_filename: bool = False,
    ):
        self.images_dir    = Path(images_dir)
        self.masks_dir     = Path(masks_dir) if masks_dir else None
        self.transform     = transform
        self.return_filename = return_filename

        exts = {'.png', '.jpg', '.jpeg'}
        self.image_files = sorted(
            [f for f in self.images_dir.iterdir() if f.suffix.lower() in exts]
        )

        if self.masks_dir is not None:
            # Verify masks exist for every image
            self.mask_files = [self.masks_dir / f.name for f in self.image_files]
            missing = [str(m) for m in self.mask_files if not m.exists()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} mask(s) not found, e.g.: {missing[0]}"
                )
        else:
            self.mask_files = None

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_path = self.image_files[idx]
        image = np.array(Image.open(img_path).convert('RGB'))

        if self.mask_files is not None:
            mask_raw = np.array(Image.open(self.mask_files[idx]))
            if mask_raw.ndim == 3:
                mask_raw = mask_raw[:, :, 0]   # keep first channel
            mask = remap_mask(mask_raw)

            if self.transform:
                aug = self.transform(image=image, mask=mask)
                image = aug['image']   # Tensor [3, H, W]
                mask  = aug['mask'].long()   # Tensor [H, W]

            if self.return_filename:
                return image, mask, img_path.name
            return image, mask

        else:
            # Test mode — no mask
            if self.transform:
                aug = self.transform(image=image)
                image = aug['image']
            if self.return_filename:
                return image, img_path.name
            return image


# --------------------------------------------------------------------------- #
# Class-weight computation (call once before training)
# --------------------------------------------------------------------------- #
def compute_class_weights(
    masks_dir: str,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from all training masks.

    Returns a float32 tensor of shape [num_classes].
    """
    masks_path = Path(masks_dir)
    pixel_counts = np.zeros(num_classes, dtype=np.float64)

    mask_files = sorted(
        [f for f in masks_path.iterdir() if f.suffix.lower() == '.png']
    )
    print(f"Computing class weights from {len(mask_files)} masks ...")
    for mf in mask_files:
        mask_raw = np.array(Image.open(mf))
        if mask_raw.ndim == 3:
            mask_raw = mask_raw[:, :, 0]
        remapped = remap_mask(mask_raw)
        for c in range(num_classes):
            pixel_counts[c] += (remapped == c).sum()

    total = pixel_counts.sum()
    # Inverse-frequency: weight = total / (num_classes * count)
    # Avoid division by zero for classes not present
    weights = np.where(
        pixel_counts > 0,
        total / (num_classes * pixel_counts),
        0.0,
    )
    # Normalize so mean weight = 1
    if weights.sum() > 0:
        weights = weights / weights.mean()

    print("Class pixel counts:")
    for i, (name, cnt, w) in enumerate(zip(CLASS_NAMES, pixel_counts, weights)):
        print(f"  [{i}] {name:<18} pixels={int(cnt):>12,}   weight={w:.4f}")

    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# DataLoader factory
# --------------------------------------------------------------------------- #
def build_dataloaders(cfg: dict, train_tfm, val_tfm):
    """Build train and val DataLoaders from config dict."""
    train_ds = OffRoadDataset(
        images_dir=cfg['data']['train_images'],
        masks_dir=cfg['data']['train_masks'],
        transform=train_tfm,
    )
    val_ds = OffRoadDataset(
        images_dir=cfg['data']['val_images'],
        masks_dir=cfg['data']['val_masks'],
        transform=val_tfm,
    )

    num_workers = cfg['training'].get('num_workers', 4)
    pin_memory  = cfg['training'].get('pin_memory', True)
    batch_size  = cfg['training']['batch_size']

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader
