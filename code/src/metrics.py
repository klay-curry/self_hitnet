from __future__ import annotations

import numpy as np
import torch


def compute_iou(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    pred_bin = (pred >= threshold).float()
    gt_bin = (gt >= 0.5).float()
    inter = (pred_bin * gt_bin).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + gt_bin.sum(dim=(1, 2, 3)) - inter
    return inter / (union + 1e-6)


def compute_mae(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.abs(pred - gt).mean(dim=(1, 2, 3))


def batch_metrics(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5):
    with torch.no_grad():
        iou = compute_iou(pred, gt, threshold)
        mae = compute_mae(pred, gt)
    return iou.mean().item(), mae.mean().item()


def compute_iou_mae(pred: torch.Tensor, gt: torch.Tensor):
    """Return mean IoU/MAE for a batch (pred, gt in [B,1,H,W])."""
    return batch_metrics(pred, gt)


def _normalize(pred: np.ndarray) -> np.ndarray:
    p_min, p_max = pred.min(), pred.max()
    if p_max - p_min < 1e-6:
        return np.zeros_like(pred)
    return (pred - p_min) / (p_max - p_min)


def compute_f_measure(pred: np.ndarray, gt: np.ndarray, beta2: float = 0.3) -> float:
    pred = _normalize(pred)
    gt_bin = gt > 0.5
    pred_bin = pred > 0.5

    tp = np.logical_and(pred_bin, gt_bin).sum()
    fp = np.logical_and(pred_bin, np.logical_not(gt_bin)).sum()
    fn = np.logical_and(np.logical_not(pred_bin), gt_bin).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f_beta = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)
    return float(f_beta)


def compute_s_measure(pred: np.ndarray, gt: np.ndarray, alpha: float = 0.5) -> float:
    """Lightweight S-measure approximation mixing structure (SSIM-like) and object terms."""
    pred = _normalize(pred)
    gt_bin = gt > 0.5
    y = gt_bin.mean()
    if y == 0:
        return float(1.0 - pred.mean())
    if y == 1:
        return float(pred.mean())

    # object-aware score
    fg = pred[gt_bin]
    bg = pred[~gt_bin]
    o_fg = fg.mean() if fg.size else 0.0
    o_bg = 1.0 - bg.mean() if bg.size else 0.0
    object_score = y * o_fg + (1 - y) * o_bg

    # region-aware (simple global SSIM-like)
    mu_x, mu_y = pred.mean(), gt_bin.mean()
    sigma_x, sigma_y = pred.var(), gt_bin.var()
    cov = ((pred - mu_x) * (gt_bin - mu_y)).mean()
    C1, C2 = 0.01, 0.03
    ssim = (2 * mu_x * mu_y + C1) * (2 * cov + C2) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2) + 1e-8)

    return float(alpha * ssim + (1 - alpha) * object_score)


def compute_e_measure(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bin = (_normalize(pred) > 0.5).astype(float)
    gt_bin = (gt > 0.5).astype(float)
    if gt_bin.sum() == 0:
        return float(1.0 - pred_bin.mean())
    align = 1.0 - np.abs(pred_bin - gt_bin).mean()
    return float(align)


def get_all_metrics(pred_tensor: torch.Tensor, gt_tensor: torch.Tensor) -> dict[str, float]:
    """Compute IoU, MAE, F-beta, S-measure, E-measure for a single-item batch."""
    pred_np = pred_tensor.squeeze().detach().cpu().numpy()
    gt_np = gt_tensor.squeeze().detach().cpu().numpy()

    iou, mae = batch_metrics(pred_tensor, gt_tensor)
    f_beta = compute_f_measure(pred_np, gt_np)
    s_measure = compute_s_measure(pred_np, gt_np)
    e_measure = compute_e_measure(pred_np, gt_np)

    return {
        "IoU": float(iou),
        "MAE": float(mae),
        "F_beta": float(f_beta),
        "S_measure": float(s_measure),
        "E_measure": float(e_measure),
    }
