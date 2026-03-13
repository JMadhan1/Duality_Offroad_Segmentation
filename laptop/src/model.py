"""
model.py — Model factory for semantic segmentation.

Supported architectures:
  1. DeepLabV3+  (segmentation_models_pytorch)
     backbones: resnet101, resnet50, efficientnet-b4, resnext101_32x8d, ...
  2. SegFormer   (HuggingFace transformers)
     variants:  nvidia/mit-b2, nvidia/mit-b3, nvidia/mit-b4

Both return a common interface:
  model(images: [B,3,H,W])  →  logits: [B, num_classes, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# DeepLabV3+  (smp)
# --------------------------------------------------------------------------- #
def build_deeplabv3plus(
    backbone: str = 'resnet101',
    encoder_weights: str = 'imagenet',
    num_classes: int = 10,
) -> nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError("Run: pip install segmentation-models-pytorch")

    model = smp.DeepLabV3Plus(
        encoder_name=backbone,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )
    return model


# --------------------------------------------------------------------------- #
# SegFormer wrapper  (HuggingFace)
# --------------------------------------------------------------------------- #
class SegFormerWrapper(nn.Module):
    """Wraps HuggingFace SegFormer so that forward() returns full-resolution
    logits [B, num_classes, H, W] matching the input image dimensions."""

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        out = self.model(pixel_values=x)
        logits = out.logits   # [B, num_classes, H/4, W/4]
        # Upsample to input resolution
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        return logits


def build_segformer(
    variant: str = 'nvidia/mit-b3',
    num_classes: int = 10,
) -> nn.Module:
    try:
        from transformers import SegformerForSemanticSegmentation, SegformerConfig
    except ImportError:
        raise ImportError("Run: pip install transformers")

    config = SegformerConfig.from_pretrained(variant)
    config.num_labels   = num_classes
    config.id2label     = {i: str(i) for i in range(num_classes)}
    config.label2id     = {str(i): i for i in range(num_classes)}

    model = SegformerForSemanticSegmentation.from_pretrained(
        variant,
        config=config,
        ignore_mismatched_sizes=True,
    )
    return SegFormerWrapper(model)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_model(cfg: dict) -> nn.Module:
    """Build model from config dict.

    cfg['model']['architecture'] : 'deeplabv3plus' | 'segformer'
    """
    model_cfg = cfg['model']
    arch = model_cfg['architecture'].lower()
    num_classes = cfg['classes']['num_classes']

    if arch == 'deeplabv3plus':
        model = build_deeplabv3plus(
            backbone=model_cfg.get('backbone', 'resnet101'),
            encoder_weights=model_cfg.get('encoder_weights', 'imagenet'),
            num_classes=num_classes,
        )
        print(f"[Model] DeepLabV3+  backbone={model_cfg.get('backbone', 'resnet101')}")
    elif arch == 'segformer':
        variant = model_cfg.get('segformer_variant', 'nvidia/mit-b3')
        model = build_segformer(variant=variant, num_classes=num_classes)
        print(f"[Model] SegFormer  variant={variant}")
    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose 'deeplabv3plus' or 'segformer'.")

    # Parameter count
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Parameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")
    return model


# --------------------------------------------------------------------------- #
# Multi-scale inference helper (used in TTA)
# --------------------------------------------------------------------------- #
def multiscale_predict(
    model: nn.Module,
    image: torch.Tensor,
    scales: list = (0.75, 1.0, 1.25),
    flip: bool = True,
) -> torch.Tensor:
    """Run inference at multiple scales and average softmax probabilities.

    Args:
        model:  segmentation model in eval mode.
        image:  [1, 3, H, W]  normalised tensor (single image).
        scales: list of scale factors.
        flip:   if True, also predict on horizontally flipped image.

    Returns:
        averaged softmax probabilities [1, C, H, W] at original H×W.
    """
    _, _, H, W = image.shape
    accumulated: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for scale in scales:
            sh = int(round(H * scale / 32)) * 32
            sw = int(round(W * scale / 32)) * 32
            sh, sw = max(sh, 32), max(sw, 32)

            img_scaled = F.interpolate(image, size=(sh, sw), mode='bilinear', align_corners=False)

            for do_flip in ([False, True] if flip else [False]):
                inp = img_scaled.flip(-1) if do_flip else img_scaled
                logits = model(inp)   # [1, C, sh, sw]
                if do_flip:
                    logits = logits.flip(-1)
                # Upsample back to original resolution
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
                accumulated.append(torch.softmax(logits, dim=1))

    return torch.stack(accumulated, dim=0).mean(dim=0)  # [1, C, H, W]
