from __future__ import annotations

import torch
import torch.nn.functional as F


def structure_loss(pred, mask):
    """Weighted BCE + weighted IoU as used in HitNet."""
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred_sig = torch.sigmoid(pred)
    inter = ((pred_sig * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sig + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def multi_scale_loss(stage_preds, final_pred, mask, gamma: float = 0.2):
    """Aggregate loss for HitNet multi-stage outputs."""
    losses = [structure_loss(out, mask) for out in stage_preds]
    loss_p1 = sum((gamma * idx) * l for idx, l in enumerate(losses))
    loss_p2 = structure_loss(final_pred, mask)
    return loss_p1 + loss_p2
