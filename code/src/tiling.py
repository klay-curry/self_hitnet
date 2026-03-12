from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .utils import mask_to_bbox


@torch.no_grad()
def infer_single(model: torch.nn.Module, image: np.ndarray, device: torch.device, size: int = 352) -> np.ndarray:
    img = cv2.resize(image, (size, size))
    tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    tensor = (tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)
    tensor = tensor.unsqueeze(0).to(device)
    stage_preds, final_pred = model(tensor)
    pred = (stage_preds[-1] + final_pred).sigmoid().squeeze(0).squeeze(0).cpu().numpy()
    pred = cv2.resize(pred, (image.shape[1], image.shape[0]))
    return pred


def _iter_patches(img: np.ndarray, patch: int, stride: int):
    h, w = img.shape[:2]
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1, x1 = y, x
            y2, x2 = min(y + patch, h), min(x + patch, w)
            yield x1, y1, x2, y2


@torch.no_grad()
def infer_tiled(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    patch: int = 352,
    stride: int = 320,
) -> np.ndarray:
    h, w = image.shape[:2]
    prob = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for x1, y1, x2, y2 in _iter_patches(image, patch, stride):
        crop = image[y1:y2, x1:x2, :]
        tensor = torch.from_numpy(cv2.resize(crop, (patch, patch))).float().permute(2, 0, 1) / 255.0
        tensor = (tensor - mean) / std
        tensor = tensor.unsqueeze(0).to(device)
        stage_preds, final_pred = model(tensor)
        pred = (stage_preds[-1] + final_pred).sigmoid().squeeze(0).squeeze(0).cpu().numpy()
        pred = cv2.resize(pred, (x2 - x1, y2 - y1))
        prob[y1:y2, x1:x2] += pred
        count[y1:y2, x1:x2] += 1

    prob /= np.maximum(count, 1e-6)
    return prob


def predict_with_multi_scale(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    device: torch.device,
    sizes: Iterable[int],
) -> Tuple[np.ndarray, dict]:
    """Predict masks for multiple target sizes, using tiling for large sizes."""
    results = {}
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    for size in sizes:
        if size <= 512:
            pred = infer_single(model, rgb, device, size=size)
        elif size <= 1024:
            pred = infer_single(model, rgb, device, size=size)
        else:
            pred = infer_tiled(model, rgb, device, patch=352, stride=320)
        results[size] = pred
    return results, {s: mask_to_bbox((pred >= 0.5).astype(np.uint8)) for s, pred in results.items()}
