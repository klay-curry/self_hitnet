from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make HitNet importable
REPO_ROOT = Path(__file__).resolve().parent.parent
"""Training script for HitNet on CAMO."""

# Make repo root and code package importable
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "code"))
HITNET_ROOT = REPO_ROOT / "HitNet"

from HitNet.lib.pvt import Hitnet  # type: ignore  # noqa: E402
from src.dataset import build_loader  # type: ignore  # noqa: E402
from src.losses import multi_scale_loss  # type: ignore  # noqa: E402
from src.metrics import batch_metrics  # type: ignore  # noqa: E402
from src.utils import save_checkpoint, seed_everything  # type: ignore  # noqa: E402
from src.logger_utils import get_timestamp, setup_logger  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HitNet on CAMO for COD")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=352)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(REPO_ROOT / "code" / "assert" / "data" / "CAMO" / "CAMO-D"),
    )
    parser.add_argument("--save-dir", type=str, default=str(REPO_ROOT / "code" / "checkpoints"))
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (REPO_ROOT / data_root).resolve()
    train_images = data_root / "train"
    train_masks = data_root / "gt"
    val_images = data_root / "test"
    val_masks = data_root / "gt"

    train_loader = build_loader(
        root_dir=str(data_root),
        image_dir=str(train_images),
        mask_dir=str(train_masks),
        mode="train",
        size=args.train_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = build_loader(
        root_dir=str(data_root),
        image_dir=str(val_images),
        mask_dir=str(val_masks),
        mode="val",
        size=args.train_size,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
    )
    return train_loader, val_loader


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    iou_meter, mae_meter, count = 0.0, 0.0, 0
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)
            stage_preds, final_pred = model(images)
            pred = (stage_preds[-1] + final_pred).sigmoid()
            iou, mae = batch_metrics(pred, masks)
            batch = images.size(0)
            iou_meter += iou * batch
            mae_meter += mae * batch
            count += batch
    model.train()
    return iou_meter / count, mae_meter / count


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    timestamp = get_timestamp()
    log_dir = REPO_ROOT / "code" / "log" / "train"
    logger, _ = setup_logger(log_dir, "train", timestamp)
    logger.info(f"Start training session {timestamp}")
    logger.info(f"Args: {args}")

    train_loader, val_loader = get_dataloaders(args)

    model = Hitnet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_iou = 0.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / f"{timestamp}_hitnet_best.pth"
    final_path = save_dir / f"{timestamp}_hitnet_last.pth"

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        steps = 0
        for images, masks, _ in pbar:
            images = images.to(device)
            masks = masks.to(device)

            stage_preds, final_pred = model(images)
            loss = multi_scale_loss(stage_preds, final_pred, masks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            running_loss += loss.item()
            steps += 1

        avg_loss = running_loss / max(steps, 1)
        logger.info(f"Epoch {epoch}/{args.epochs} - train_loss={avg_loss:.4f}")

        if epoch % args.val_every == 0:
            iou, mae = evaluate(model, val_loader, device)
            logger.info(f"[Val] Epoch {epoch}: IoU={iou:.4f}, MAE={mae:.4f}")
            if iou > best_iou:
                best_iou = iou
                save_checkpoint(model, best_path)
                logger.info(f"Saved new best to {best_path}")

    # save final
    save_checkpoint(model, final_path)
    logger.info(f"Training complete. Best IoU={best_iou:.4f}. Final weights at {final_path}")


if __name__ == "__main__":
    main()
