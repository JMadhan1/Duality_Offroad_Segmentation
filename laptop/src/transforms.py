"""
transforms.py — Albumentations augmentation pipelines (OPTIMIZED for speed).

Compatible with albumentations >= 2.0.8

Strategy:
  Training  -> lightweight but effective augmentations (no CPU bottleneck)
  Val/Test  -> resize + normalize only
  TTA       -> multi-scale + horizontal flip (averaged at inference time)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_transforms(input_size=(384, 384)):
    """Fast but effective augmentations for training.

    Heavy transforms (ElasticTransform, GridDistortion, RandomSunFlare,
    RandomShadow, RandomFog) removed -- they bottleneck CPU and slow training.
    The kept transforms are the most impactful for domain generalisation.
    """
    h, w = input_size
    return A.Compose([
        # --- Spatial / geometric (fast) ---
        A.RandomResizedCrop(
            size=(h, w),          # alb 2.0 API: size=(H, W)
            scale=(0.5, 1.0),
            ratio=(0.75, 1.33),
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.15),
        A.Rotate(limit=10, fill=0, fill_mask=255, p=0.3),

        # --- Color / photometric (fast, image only) ---
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.4),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.08, p=0.3),

        # --- Noise / blur (lightweight) ---
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # --- Occlusion simulation (helps rare classes) ---
        A.CoarseDropout(
            num_holes_range=(1, 4),             # alb 2.0 API
            hole_height_range=(h // 32, h // 16),
            hole_width_range=(w // 32, w // 16),
            fill=0,
            fill_mask=255,
            p=0.2,
        ),

        # --- Final normalisation ---
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(input_size=(384, 384)):
    """Minimal transforms for validation: resize + normalize."""
    h, w = input_size
    return A.Compose([
        A.Resize(height=h, width=w),
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ])


def get_test_transforms(input_size=(384, 384)):
    """Same as val transforms -- used for test set inference."""
    return get_val_transforms(input_size)


# --------------------------------------------------------------------------- #
# Test-Time Augmentation helpers
# --------------------------------------------------------------------------- #
def get_tta_transforms(input_size=(384, 384), scales=(0.75, 1.0, 1.25)):
    """Return list of (name, transform) for TTA inference.

    Each transform resizes to a scaled version of input_size and normalizes.
    The caller must handle upsampling predictions back to the original size.
    """
    h, w = input_size
    tfms = []
    for scale in scales:
        sh, sw = int(h * scale), int(w * scale)
        # Make dimensions divisible by 32
        sh = max((sh // 32) * 32, 32)
        sw = max((sw // 32) * 32, 32)
        for flip in [False, True]:
            ops = [A.Resize(height=sh, width=sw)]
            if flip:
                ops.append(A.HorizontalFlip(p=1.0))
            ops += [A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD), ToTensorV2()]
            name = f"scale{scale:.2f}_{'flip' if flip else 'orig'}"
            tfms.append((name, A.Compose(ops)))
    return tfms


def denormalize(tensor):
    """Undo ImageNet normalization; returns tensor in [0, 1]."""
    mean = tensor.new_tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std  = tensor.new_tensor(_IMAGENET_STD).view(3, 1, 1)
    return tensor * std + mean
