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

# ---------- metadata & split ----------
def _read_metadata(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "isic_id" not in df.columns and "image_name" in df.columns:
        df = df.rename(columns={"image_name":"isic_id"})
    assert "isic_id" in df.columns and "target" in df.columns
    keep = ["isic_id","target"] + (["patient_id"] if "patient_id" in df.columns else [])
    df = df[keep].dropna(subset=["isic_id","target"]).reset_index(drop=True)
    df["isic_id"] = df["isic_id"].astype(str)
    df["target"]  = df["target"].astype(int)
    return df

def _stratified_split(df, val_frac, test_frac, seed):
    rng = random.Random(seed)
    parts = []
    for _, g in df.groupby("target"):
        idxs = list(g.index); rng.shuffle(idxs)
        n = len(idxs); n_test = int(round(test_frac*n)); n_val = int(round(val_frac*n))
        test_idx = idxs[:n_test]; val_idx = idxs[n_test:n_test+n_val]; train_idx = idxs[n_test+n_val:]
        parts += [("train", g.loc[train_idx]), ("val", g.loc[val_idx]), ("test", g.loc[test_idx])]
    train = pd.concat([p for k,p in parts if k=="train"]).sample(frac=1, random_state=seed)
    val   = pd.concat([p for k,p in parts if k=="val"]).sample(frac=1, random_state=seed)
    test  = pd.concat([p for k,p in parts if k=="test"]).sample(frac=1, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def _grouped_split(df, group_col, val_frac, test_frac, seed):
    rng = random.Random(seed)
    parts = []
    for _, g in df.groupby("target"):
        groups = list(g[group_col].dropna().astype(str).unique())
        rng.shuffle(groups)
        n = len(groups); n_test = int(round(test_frac*n)); n_val = int(round(val_frac*n))
        test_g = set(groups[:n_test]); val_g = set(groups[n_test:n_test+n_val]); train_g = set(groups[n_test+n_val:])
        train = g[g[group_col].astype(str).isin(train_g)]
        val   = g[g[group_col].astype(str).isin(val_g)]
        test  = g[g[group_col].astype(str).isin(test_g)]
        parts += [("train", train), ("val", val), ("test", test)]
    train = pd.concat([p for k,p in parts if k=="train"]).sample(frac=1, random_state=seed)
    val   = pd.concat([p for k,p in parts if k=="val"]).sample(frac=1, random_state=seed)
    test  = pd.concat([p for k,p in parts if k=="test"]).sample(frac=1, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ---------- datasets ----------
class ISICSingle(Dataset):
    def __init__(self, images_dir: str, table: pd.DataFrame, train: bool):
        self.dir = images_dir
        self.tfm = _build_transforms(train)
        seed = getattr(config, "SEED", 42)
        lim = int(getattr(config, "FAST_LIMIT_PER_CLASS", 0))
        if getattr(config, "FAST_DEBUG", False) and lim > 0:
            frames = []
            for _, g in table.groupby("target"):
                frames.append(g.sample(n=min(lim, len(g)), random_state=seed))
            table = pd.concat(frames, axis=0).sample(frac=1, random_state=seed).reset_index(drop=True)
        keep = []
        for _, r in table.iterrows():
            p = _resolve_path(images_dir, str(r["isic_id"]))
            if p: keep.append((p, int(r["target"])))
        self.paths  = [p for p,_ in keep]
        self.labels = [y for _,y in keep]

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        x = self.tfm(img)
        y = int(self.labels[idx])
        return x, torch.tensor(y, dtype=torch.long)
