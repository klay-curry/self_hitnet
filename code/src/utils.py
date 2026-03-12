from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, checkpoint: Path, map_location=None) -> torch.nn.Module:
    state = torch.load(checkpoint, map_location=map_location)
    model.load_state_dict(state)
    return model


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    mask_bin = (mask > 0.5).astype(np.uint8)
    overlay = image.copy()
    overlay[mask_bin == 1] = (0, 0, 255)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def mask_to_bbox(mask_binary: np.ndarray, min_area: int = 50) -> List[Tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= min_area:
            bboxes.append((x, y, x + w, y + h))
    return bboxes


def draw_bboxes(image: np.ndarray, bboxes: Iterable[Tuple[int, int, int, int]], color=(0, 0, 255)) -> np.ndarray:
    out = image.copy()
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    return out


def generate_comparison(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """Create a color-coded comparison between prediction and GT.

    Colors (on white background):
    - Green: true positive (pred=1, gt=1)
    - Red: false negative (pred=0, gt=1)
    - Blue: false positive (pred=1, gt=0)
    """
    pred = (pred_mask > 0.5).astype(np.uint8)
    gt = (gt_mask > 0.5).astype(np.uint8)

    h, w = pred.shape
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    tp = (pred == 1) & (gt == 1)
    fn = (pred == 0) & (gt == 1)
    fp = (pred == 1) & (gt == 0)

    canvas[tp] = [0, 255, 0]      # green
    canvas[fn] = [0, 0, 255]      # red
    canvas[fp] = [255, 0, 0]      # blue

    return canvas


def add_title(image: np.ndarray, text: str) -> np.ndarray:
    """Add a top white band with a text label."""
    h, w = image.shape[:2]
    pad_h = 40
    img_padded = cv2.copyMakeBorder(image, pad_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    font_scale = max(0.5, w / 600.0)
    thickness = max(1, int(w / 300.0))
    cv2.putText(img_padded, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return img_padded


def generate_analysis_strip(
    image_bgr: np.ndarray,
    gt_mask: np.ndarray | None,
    pred_bin: np.ndarray,
    pred_heatmap: np.ndarray,
) -> np.ndarray:
    """Create a four-panel strip: [image | GT | pred mask | heatmap]."""
    vis_img = image_bgr.copy()

    if gt_mask is not None:
        gt_disp = gt_mask
        if gt_disp.max() <= 1:
            gt_disp = gt_disp * 255
        vis_gt = cv2.cvtColor(gt_disp.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        vis_gt = np.zeros_like(vis_img)
        cv2.putText(
            vis_gt,
            "No GT",
            (10, vis_gt.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    vis_pred = cv2.cvtColor(pred_bin.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    heatmap_uint8 = (np.clip(pred_heatmap, 0, 1) * 255).astype(np.uint8)
    vis_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    final_img = add_title(vis_img, "Original Image")
    final_gt = add_title(vis_gt, "Ground Truth")
    final_pred = add_title(vis_pred, "Prediction Mask")
    final_heat = add_title(vis_heatmap, "Saliency Map")

    return np.concatenate([final_img, final_gt, final_pred, final_heat], axis=1)
