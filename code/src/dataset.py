from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader


class CODDataset(Dataset):
    """
    Camouflaged object dataset supporting two modes:
    - Directory mode: image_dir + mask_dir (fast for CAMO / Camouflage-people).
    - JSON mode: COCO-style json_file + root_dir (for NC4K where JSON provides names).
    """

    def __init__(
        self,
        root_dir: str,
        json_file: Optional[str] = None,
        image_dir: Optional[str] = None,
        mask_dir: Optional[str] = None,
        mode: str = "train",
        size: int = 352,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.size = size
        self.use_json = json_file is not None

        if self.use_json:
            self.coco = COCO(json_file)
            self.ids = list(self.coco.imgs.keys())
            self.root_dir = Path(root_dir)
        else:
            if image_dir is None or mask_dir is None:
                raise ValueError("image_dir and mask_dir are required when json_file is None")
            self.image_paths = [
                os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))
            ]
            self.mask_dir = Path(mask_dir)

        self.transform_train = A.Compose(
            [
                A.Resize(size, size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        self.transform_val = A.Compose(
            [
                A.Resize(size, size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:  # pragma: no cover - simple proxy
        return len(self.ids) if self.use_json else len(self.image_paths)

    def __getitem__(self, index: int):
        if self.use_json:
            img_id = self.ids[index]
            img_meta = self.coco.loadImgs(img_id)[0]
            image_path = self.root_dir / img_meta["file_name"]
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            filename = img_meta["file_name"]
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            image_path = self.image_paths[index]
            filename = os.path.basename(image_path)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask_name = filename.rsplit(".", 1)[0] + ".png"
            mask_path = self.mask_dir / mask_name
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), 0)
                _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
            else:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)

        transform = self.transform_train if self.mode == "train" else self.transform_val
        augmented = transform(image=image, mask=mask)
        img_tensor = augmented["image"]
        mask_tensor = augmented["mask"].float().unsqueeze(0)
        return img_tensor, mask_tensor, filename


def build_loader(
    *,
    root_dir: str,
    json_file: Optional[str] = None,
    image_dir: Optional[str] = None,
    mask_dir: Optional[str] = None,
    mode: str,
    size: int,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    dataset = CODDataset(
        root_dir=root_dir,
        json_file=json_file,
        image_dir=image_dir,
        mask_dir=mask_dir,
        mode=mode,
        size=size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if mode == "train" else False,
        num_workers=num_workers,
        pin_memory=True,
    )
