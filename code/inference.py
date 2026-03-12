from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
import sys
from typing import Iterable, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
# Add repo root and code dir so packages resolve regardless of cwd
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "code"))
HITNET_ROOT = REPO_ROOT / "HitNet"

from HitNet.lib.pvt import Hitnet  # type: ignore  # noqa: E402
from src.dataset import build_loader  # type: ignore  # noqa: E402
from src.metrics import batch_metrics, get_all_metrics  # type: ignore  # noqa: E402
from src.tiling import predict_with_multi_scale, infer_tiled  # type: ignore  # noqa: E402
from src.utils import (
    draw_bboxes,
    load_model,
    mask_to_bbox,
    overlay_mask,
    generate_comparison,
    generate_analysis_strip,
)  # type: ignore  # noqa: E402
from src.logger_utils import get_timestamp, setup_logger  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for COD tasks 1-6")
    parser.add_argument("--task", type=int, choices=[1, 2, 3, 4, 5, 6], default=1)
    parser.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "code" / "checkpoints" / "hitnet_best.pth"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-root", type=str, default=str(REPO_ROOT / "code" / "answer"), help="Base directory to store inference outputs")
    parser.add_argument("--sample", type=int, default=10, help="number of samples to visualize (tasks 1-3)")
    parser.add_argument("--size", type=int, default=352, help="Input resolution for inference (e.g., 352 or 704)")
    return parser.parse_args()


def load_model_from_ckpt(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = Hitnet().to(device)
    load_model(model, checkpoint, map_location=device)
    model.eval()
    return model


@torch.no_grad()
def predict_mask(model: torch.nn.Module, image_bgr: np.ndarray, device: torch.device, size: int = 352, tta: bool = False):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def _forward(img: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(cv2.resize(img, (size, size))).float().permute(2, 0, 1) / 255.0
        tensor = (tensor - mean) / std
        tensor = tensor.unsqueeze(0).to(device)
        stage_preds, final_pred = model(tensor)
        pred = (stage_preds[-1] + final_pred).sigmoid().squeeze(0).squeeze(0).cpu().numpy()
        return cv2.resize(pred, (w, h))

    pred = _forward(rgb)
    if tta:
        flipped = _forward(np.ascontiguousarray(rgb[:, ::-1, :]))[:, ::-1]
        pred = (pred + flipped) / 2
    return pred


def extract_model_tag(ckpt_path: Path) -> str:
    """Extract timestamp-like tag from checkpoint filename for organizing outputs."""
    m = re.search(r"(\d{8}_\d{4})", ckpt_path.stem)
    if m:
        return m.group(1)
    return ckpt_path.stem


def load_gt(mask_dir: Path, name: str) -> np.ndarray | None:
    """Load GT mask. Try original name, then swap to .png if needed."""
    path = mask_dir / name
    if not path.exists():
        alt_name = Path(name).with_suffix(".png").name
        path = mask_dir / alt_name
        if not path.exists():
            return None

    gt = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return None
    return (gt > 127).astype(np.uint8)


def save_result(
    name: str,
    image_bgr: np.ndarray,
    pred_mask: np.ndarray,
    out_base: Path,
    gt_mask: np.ndarray | None = None,
    per_image_dir: bool = False,
    metrics: dict[str, float] | None = None,
):
    """Save overlay, binary mask, analysis strip, comparisons, and optional metrics."""
    h, w = image_bgr.shape[:2]
    pred_resized = cv2.resize(pred_mask, (w, h))
    pred_bin = (pred_resized >= 0.5).astype(np.uint8) * 255

    gt_resized = None
    if gt_mask is not None:
        gt_resized = cv2.resize(gt_mask, (w, h))

    if per_image_dir:
        img_dir = out_base / Path(name).stem
        img_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = img_dir / "overlay.png"
        pred_path = img_dir / "pred_mask.png"
        compare_path = img_dir / "compare_gt.png"
        analysis_path = img_dir / "analysis.jpg"
        metrics_path = img_dir / "metrics.txt"
    else:
        overlay_dir = out_base / "overlay"
        pred_dir = out_base / "prediction_gt"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / name
        pred_path = pred_dir / name
        compare_path = None
        analysis_path = None
        metrics_path = None
        if gt_mask is not None:
            compare_dir = out_base / "compare_gt"
            compare_dir.mkdir(parents=True, exist_ok=True)
            compare_path = compare_dir / name

    overlay = overlay_mask(image_bgr, pred_resized)
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(pred_path), pred_bin)

    if gt_resized is not None and compare_path is not None:
        comp = generate_comparison(pred_resized, gt_resized)
        cv2.imwrite(str(compare_path), comp)

    if per_image_dir and analysis_path is not None:
        analysis_strip = generate_analysis_strip(image_bgr, gt_resized, pred_bin, pred_resized)
        cv2.imwrite(str(analysis_path), analysis_strip)

    if per_image_dir and metrics_path is not None and metrics:
        with open(metrics_path, "w", encoding="utf-8") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v:.6f}\n")


def run_task1(model: torch.nn.Module, device: torch.device, save_root: Path, logger, sample: int, size: int):
    """Task1: use provided question images; optionally load GT from gt subfolder."""
    input_dir = REPO_ROOT / "code" / "assert" / "问题一：可视化展示图像"
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    gt_dir = input_dir / "gt"
    has_gt = gt_dir.exists()

    task_root = save_root / "Question1"
    images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
    logger.info(f"[Task1] Found {len(images)} images in {input_dir}")

    for img_path in tqdm(images, desc="Task 1 Inference"):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            logger.warning(f"[Task1] Could not read {img_path}")
            continue

        pred = predict_mask(model, bgr, device, size=size)

        gt_np = None
        metrics = None
        if has_gt:
            gt_np = load_gt(gt_dir, img_path.name)
            if gt_np is not None:
                pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
                gt_t = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).float()
                metrics = get_all_metrics(pred_t, gt_t)

        save_result(img_path.name, bgr, pred, task_root, gt_mask=gt_np, per_image_dir=False, metrics=metrics)

    logger.info(f"[Task1] Done. Results saved to {task_root}")


def run_task2(model: torch.nn.Module, device: torch.device, save_root: Path, logger, sample: int, size: int):
    """Task2: multi-scale; optionally load GT from gt subfolder."""
    input_dir = REPO_ROOT / "code" / "assert" / "问题二：可视化展示图像"
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    gt_dir = input_dir / "gt"
    has_gt = gt_dir.exists()

    task_root = save_root / "Question2"
    images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}])
    logger.info(f"[Task2] Found {len(images)} images")

    for img_path in tqdm(images, desc="Task 2 Inference"):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            logger.warning(f"[Task2] Could not read {img_path}")
            continue

        size_match = re.search(r"_(\d{3,4})\.", img_path.name)
        target_size = int(size_match.group(1)) if size_match else size

        if target_size > 1024:
            stride = max(1, int(size * 0.9))
            pred = infer_tiled(model, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), device, patch=size, stride=stride)
        else:
            pred = predict_mask(model, bgr, device, size=target_size)

        gt_np = None
        metrics = None
        if has_gt:
            gt_np = load_gt(gt_dir, img_path.name)
            if gt_np is not None:
                pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
                gt_t = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).float()
                if pred_t.shape == gt_t.shape:
                    metrics = get_all_metrics(pred_t, gt_t)

        save_result(img_path.name, bgr, pred, task_root, gt_mask=gt_np, per_image_dir=False, metrics=metrics)

    logger.info(f"[Task2] Done. Results saved to {task_root}")


def run_task3(model: torch.nn.Module, device: torch.device, save_root: Path, logger, sample: int, size: int):
    """Task3: visualization with TTA; optional GT if provided in gt subfolder."""
    input_dir = REPO_ROOT / "code" / "assert" / "问题三：可视化展示图像"
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    gt_dir = input_dir / "gt"
    has_gt = gt_dir.exists()

    task_root = save_root / "Question3"
    bbox_dir = task_root / "overlay_bbox"
    bbox_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}])
    logger.info(f"[Task3] Found {len(images)} images")

    for img_path in tqdm(images, desc="Task 3 Inference"):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            logger.warning(f"[Task3] Could not read {img_path}")
            continue

        pred = predict_mask(model, bgr, device, size=size, tta=True)

        gt_np = None
        metrics = None
        if has_gt:
            gt_np = load_gt(gt_dir, img_path.name)
            if gt_np is not None:
                pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
                gt_t = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).float()
                metrics = get_all_metrics(pred_t, gt_t)

        save_result(img_path.name, bgr, pred, task_root, gt_mask=gt_np, per_image_dir=False, metrics=metrics)

        mask_bin = (pred >= 0.5).astype(np.uint8)
        bboxes = mask_to_bbox(mask_bin)
        overlay_bbox = draw_bboxes(overlay_mask(bgr, pred), bboxes)
        cv2.imwrite(str(bbox_dir / img_path.name), overlay_bbox)

    logger.info(f"[Task3] Done. Results saved to {task_root}")


def run_task4(model: torch.nn.Module, device: torch.device, save_root: Path, logger, size: int):
    """Full CAMO test visualization with metrics."""
    logger.info(">>> Running Task 4: Full CAMO Test Visualization")
    camo_root = REPO_ROOT / "code" / "assert" / "data" / "CAMO" / "CAMO-D"

    loader = build_loader(
        root_dir=str(camo_root),
        image_dir=str(camo_root / "test"),
        mask_dir=str(camo_root / "gt"),
        mode="val",
        size=size,
        batch_size=1,
        num_workers=2,
        shuffle=False,
    )

    task_root = save_root / "CAMO_Full"

    metrics_sum = {"IoU": 0.0, "MAE": 0.0, "F_beta": 0.0, "S_measure": 0.0, "E_measure": 0.0}
    count = 0
    for images, masks, names in tqdm(loader, desc="Processing CAMO"):
        images = images.to(device)
        masks = masks.to(device)

        stage_preds, final_pred = model(images)
        pred = (stage_preds[-1] + final_pred).sigmoid()

        current_metrics = get_all_metrics(pred, masks)
        for k in metrics_sum:
            metrics_sum[k] += current_metrics[k]
        count += 1

        pred_np = pred.squeeze(0).squeeze(0).detach().cpu().numpy()
        gt_np = masks.squeeze(0).squeeze(0).detach().cpu().numpy()
        img_path = camo_root / "test" / names[0]
        orig = cv2.imread(str(img_path))
        if orig is None:
            logger.info(f"[Task4] Could not read {img_path}")
            continue

        save_result(names[0], orig, pred_np, task_root, gt_mask=gt_np, per_image_dir=True, metrics=current_metrics)

    if count > 0:
        avg_metrics = {k: v / count for k, v in metrics_sum.items()}
        logger.info(
            "Task 4 Complete. Metrics: "
            + ", ".join([f"{k}={avg_metrics[k]:.4f}" for k in ["IoU", "MAE", "F_beta", "S_measure", "E_measure"]])
        )
        summary_path = task_root / "summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("Task 4 - CAMO Full Dataset Summary\n")
            f.write("==================================\n")
            for k, v in avg_metrics.items():
                f.write(f"{k}: {v:.6f}\n")
    logger.info(f"Visualizations saved to {task_root}")


def run_task5(model: torch.nn.Module, device: torch.device, save_root: Path, logger, size: int):
    """Full NC4K test: visualization + optional metrics if GT exists."""
    logger.info(">>> Running Task 5: Full NC4K Test")
    nc4k_root = REPO_ROOT / "code" / "assert" / "data" / "NC4K" / "NC4K-D" / "test"
    gt_dir = nc4k_root.parent / "gt"

    task_root = save_root / "NC4K_Full"

    metrics_sum = {"IoU": 0.0, "MAE": 0.0, "F_beta": 0.0, "S_measure": 0.0, "E_measure": 0.0}
    count = 0
    nc_imgs = sorted([p for p in nc4k_root.iterdir() if p.suffix.lower() in {".jpg", ".png"}])
    for img_path in tqdm(nc_imgs, desc="Processing NC4K"):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            logger.info(f"[Task5] Missing image {img_path}")
            continue

        pred = predict_mask(model, bgr, device, size=size, tta=True)
        gt_np = load_gt(gt_dir, img_path.name)
        current_metrics = None
        if gt_np is not None:
            pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
            gt_t = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).float()
            current_metrics = get_all_metrics(pred_t, gt_t)
            for k in metrics_sum:
                metrics_sum[k] += current_metrics[k]
            count += 1

        save_result(img_path.name, bgr, pred, task_root, gt_mask=gt_np, per_image_dir=True, metrics=current_metrics)

    if count > 0:
        avg_metrics = {k: v / count for k, v in metrics_sum.items()}
        logger.info(
            "Task 5 Complete. Metrics: "
            + ", ".join([f"{k}={avg_metrics[k]:.4f}" for k in ["IoU", "MAE", "F_beta", "S_measure", "E_measure"]])
        )
        summary_path = task_root / "summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("Task 5 - NC4K Full Dataset Summary\n")
            f.write("==================================\n")
            for k, v in avg_metrics.items():
                f.write(f"{k}: {v:.6f}\n")
    else:
        logger.info("Task 5 Complete. No GT found; metrics skipped.")
    logger.info(f"Results saved to {task_root}")


def run_task6(model: torch.nn.Module, device: torch.device, save_root: Path, logger, size: int):
    """Full Camouflage-people inference."""
    logger.info(">>> Running Task 6: Full Camouflage-people Test")
    cam_people_root = REPO_ROOT / "code" / "assert" / "data" / "Camouflage-people" / "CamouflageData"
    img_dir = cam_people_root / "img"
    gt_dir = cam_people_root / "gt"

    task_root = save_root / "Camouflage_people_Full"

    metrics_sum = {"IoU": 0.0, "MAE": 0.0, "F_beta": 0.0, "S_measure": 0.0, "E_measure": 0.0}
    count = 0
    cam_imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}])
    for img_path in tqdm(cam_imgs, desc="Processing People"):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            logger.info(f"[Task6] Missing image {img_path}")
            continue

        pred = predict_mask(model, bgr, device, size=size, tta=True)
        gt_np = load_gt(gt_dir, img_path.name)
        current_metrics = None
        if gt_np is not None:
            pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
            gt_t = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).float()
            current_metrics = get_all_metrics(pred_t, gt_t)
            for k in metrics_sum:
                metrics_sum[k] += current_metrics[k]
            count += 1

        save_result(img_path.name, bgr, pred, task_root, gt_mask=gt_np, per_image_dir=True, metrics=current_metrics)

    if count > 0:
        avg_metrics = {k: v / count for k, v in metrics_sum.items()}
        logger.info(
            "Task 6 Complete. Metrics: "
            + ", ".join([f"{k}={avg_metrics[k]:.4f}" for k in ["IoU", "MAE", "F_beta", "S_measure", "E_measure"]])
        )
        summary_path = task_root / "summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("Task 6 - Camouflage People Full Dataset Summary\n")
            f.write("===========================================\n")
            for k, v in avg_metrics.items():
                f.write(f"{k}: {v:.6f}\n")
    else:
        logger.info("Task 6 Complete. No GT found; metrics skipped.")
    logger.info(f"Results saved to {task_root}")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    model = load_model_from_ckpt(ckpt_path, device)

    log_dir = REPO_ROOT / "code" / "log" / "interface"
    logger, run_ts = setup_logger(log_dir, "interface")
    model_tag = extract_model_tag(ckpt_path)
    save_root = Path(args.save_root) / model_tag
    save_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Start inference run {run_ts} | task={args.task} | size={args.size} | checkpoint={ckpt_path} | output_root={save_root}"
    )

    if args.task == 1:
        run_task1(model, device, save_root, logger, args.sample, args.size)
    elif args.task == 2:
        run_task2(model, device, save_root, logger, args.sample, args.size)
    elif args.task == 3:
        run_task3(model, device, save_root, logger, args.sample, args.size)
    elif args.task == 4:
        run_task4(model, device, save_root, logger, args.size)
    elif args.task == 5:
        run_task5(model, device, save_root, logger, args.size)
    else:
        run_task6(model, device, save_root, logger, args.size)


if __name__ == "__main__":
    main()
