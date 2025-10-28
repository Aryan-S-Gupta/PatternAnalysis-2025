import os, random, glob
from typing import Optional, List, Dict
from collections import defaultdict

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
from torchvision.transforms import v2

import config

# ---------- transforms ----------
def _build_transforms(train: bool) -> v2.Compose:
    if getattr(config, "FAST_DEBUG", False):
        aug = [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(p=0.2),
            v2.RandomRotation(8),
        ]
    else:
        aug = [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(p=0.2),
            v2.RandomRotation(12),
            v2.ColorJitter(0.15, 0.15, 0.10, 0.05),
            v2.RandomResizedCrop((config.IMAGE_SIZE, config.IMAGE_SIZE), scale=(0.85, 1.0)),
        ]
    common = [
        v2.ToImage(),
        v2.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ]
    return v2.Compose((aug if train else []) + common)

_EXTS = (".jpg",".jpeg",".png",".JPG",".JPEG",".PNG")
def _resolve_path(images_dir: str, stem: str) -> Optional[str]:
    if getattr(config, "ASSUME_JPG", True):
        p = os.path.join(images_dir, stem + ".jpg")
        if os.path.exists(p): return p
    for ext in _EXTS:
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p): return p
    return None
