"""
losses.py — Combined loss for semantic segmentation.

Total_Loss = ce_weight * CrossEntropyLoss
           + dice_weight * DiceLoss
           + focal_weight * FocalLoss

All three handle ignore_index=255 correctly.
Class-frequency weights are supported for CrossEntropy and Focal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Dice Loss
# --------------------------------------------------------------------------- #
class DiceLoss(nn.Module):
    """Soft Dice Loss averaged over valid (non-ignore) classes.

    Directly optimises an IoU-like metric.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.num_classes   = num_classes
        self.ignore_index  = ignore_index
        self.smooth        = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C, H, W]  (raw, un-normalised)
            targets: [B, H, W]     (long, values 0..C-1 or ignore_index)
        """
        probs = F.softmax(logits, dim=1)   # [B, C, H, W]

        # Build valid mask
        valid = (targets != self.ignore_index)   # [B, H, W]

        dice_losses = []
        for c in range(self.num_classes):
            target_c = ((targets == c) & valid).float()   # [B, H, W]
            pred_c   = probs[:, c] * valid.float()        # [B, H, W]

            intersection = (pred_c * target_c).sum()
            cardinality  = pred_c.sum() + target_c.sum()

            dice = 1.0 - (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            dice_losses.append(dice)

        return torch.stack(dice_losses).mean()


# --------------------------------------------------------------------------- #
# Focal Loss
# --------------------------------------------------------------------------- #
class FocalLoss(nn.Module):
    """Focal Loss for multi-class segmentation.

    Down-weights easy examples, forcing the model to focus on hard pixels
    and minority classes (Flowers, Logs).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        ignore_index: int = 255,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.gamma        = gamma
        self.weight       = weight
        self.ignore_index = ignore_index
        self.reduction    = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard CE loss (no reduction yet)
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.weight.to(logits.device) if self.weight is not None else None,
            ignore_index=self.ignore_index,
            reduction='none',
        )   # [B, H, W]

        # p_t = exp(-ce_loss)  →  focal factor = (1 - p_t) ** gamma
        p_t   = torch.exp(-ce_loss)
        focal = (1.0 - p_t) ** self.gamma * ce_loss

        # Ignore masked pixels
        valid = (targets != self.ignore_index)
        focal = focal[valid]

        if self.reduction == 'mean':
            return focal.mean() if focal.numel() > 0 else focal.sum()
        return focal.sum()


# --------------------------------------------------------------------------- #
# Combined Loss
# --------------------------------------------------------------------------- #
class CombinedLoss(nn.Module):
    """Weighted combination of CrossEntropy + Dice + Focal.

    Default weights:  CE=0.5, Dice=0.3, Focal=0.2
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
        ignore_index: int = 255,
        ce_weight: float  = 0.5,
        dice_weight: float = 0.3,
        focal_weight: float = 0.2,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.ce_w    = ce_weight
        self.dice_w  = dice_weight
        self.focal_w = focal_weight

        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
        )
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.focal = FocalLoss(
            gamma=focal_gamma,
            weight=class_weights,
            ignore_index=ignore_index,
        )

    def to(self, device):
        super().to(device)
        if self.ce.weight is not None:
            self.ce.weight = self.ce.weight.to(device)
        if self.focal.weight is not None:
            self.focal.weight = self.focal.weight.to(device)
        return self

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """Returns total loss and a dict of individual components."""
        ce_loss    = self.ce(logits, targets)
        dice_loss  = self.dice(logits, targets)
        focal_loss = self.focal(logits, targets)

        total = self.ce_w * ce_loss + self.dice_w * dice_loss + self.focal_w * focal_loss

        components = {
            'ce':    ce_loss.item(),
            'dice':  dice_loss.item(),
            'focal': focal_loss.item(),
            'total': total.item(),
        }
        return total, components


# --------------------------------------------------------------------------- #
# CE-only loss wrapper (same interface as CombinedLoss)
# --------------------------------------------------------------------------- #
class CEOnlyLoss(nn.Module):
    """Plain CrossEntropyLoss wrapped to return (loss, components) like CombinedLoss.

    Use this during Phase 1 training to avoid Dice/Focal instability.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)

    def to(self, device):
        super().to(device)
        if self.ce.weight is not None:
            self.ce.weight = self.ce.weight.to(device)
        return self

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        loss = self.ce(logits, targets)
        components = {'ce': loss.item(), 'dice': 0.0, 'focal': 0.0, 'total': loss.item()}
        return loss, components


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_criterion(cfg: dict, class_weights: torch.Tensor | None = None):
    """Construct the loss from config.

    loss.loss_type = 'ce_only'   → CEOnlyLoss  (safe for early training)
    loss.loss_type = 'combined'  → CombinedLoss (CE + Dice + Focal)
    """
    loss_cfg     = cfg.get('loss', {})
    num_classes  = cfg['classes']['num_classes']
    ignore_index = cfg['classes']['ignore_index']

    # Apply / drop class weights
    if not loss_cfg.get('use_class_weights', True):
        class_weights = None
    elif class_weights is not None:
        # Cap extreme weights to prevent gradient spikes
        cap = float(loss_cfg.get('weight_cap', 10.0))
        class_weights = class_weights.clamp(max=cap)
        print(f"[Loss] Class weights (capped at {cap}x): {class_weights.tolist()}")

    loss_type = loss_cfg.get('loss_type', 'ce_only')

    if loss_type == 'ce_only':
        print(f"[Loss] Using CE-only loss (Phase 1 — stable training)")
        return CEOnlyLoss(class_weights=class_weights, ignore_index=ignore_index)

    print(f"[Loss] Using Combined loss  CE={loss_cfg.get('ce_weight', 0.5)}"
          f"  Dice={loss_cfg.get('dice_weight', 0.3)}  Focal={loss_cfg.get('focal_weight', 0.2)}")
    return CombinedLoss(
        num_classes=num_classes,
        class_weights=class_weights,
        ignore_index=ignore_index,
        ce_weight=loss_cfg.get('ce_weight', 0.5),
        dice_weight=loss_cfg.get('dice_weight', 0.3),
        focal_weight=loss_cfg.get('focal_weight', 0.2),
        focal_gamma=loss_cfg.get('focal_gamma', 2.0),
    )
