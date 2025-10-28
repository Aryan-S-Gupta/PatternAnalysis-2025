import os
import random
import glob
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
    aug = [
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(p=0.2),
        v2.RandomRotation(12),
        v2.ColorJitter(0.15, 0.15, 0.10, 0.05),
        v2.RandomResizedCrop(
            (config.IMAGE_SIZE, config.IMAGE_SIZE), scale=(0.85, 1.0)),
    ]
    common = [
        v2.ToImage(),
        v2.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return v2.Compose((aug if train else []) + common)


_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff",
         ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF")


def _resolve_path(images_dir: str, stem: str) -> Optional[str]:
    if getattr(config, "ASSUME_JPG", True):
        p = os.path.join(images_dir, stem + ".jpg")
        if os.path.exists(p):
            return p
    for ext in _EXTS:
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None

# ---------- metadata & split ----------


def _read_metadata(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "isic_id" not in df.columns and "image_name" in df.columns:
        df = df.rename(columns={"image_name": "isic_id"})
    assert "isic_id" in df.columns and "target" in df.columns
    keep = ["isic_id", "target"] + \
        (["patient_id"] if "patient_id" in df.columns else [])
    df = df[keep].dropna(subset=["isic_id", "target"]).reset_index(drop=True)
    df["isic_id"] = df["isic_id"].astype(str)
    df["target"] = df["target"].astype(int)
    return df


def _stratified_split(df, val_frac, test_frac, seed):
    rng = random.Random(seed)
    parts = []
    for _, g in df.groupby("target"):
        idxs = list(g.index)
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(round(test_frac * n))
        n_val = int(round(val_frac * n))
        test_idx = idxs[:n_test]
        val_idx = idxs[n_test:n_test + n_val]
        train_idx = idxs[n_test + n_val:]
        parts += [("train", g.loc[train_idx]),
                  ("val", g.loc[val_idx]), ("test", g.loc[test_idx])]
    train = pd.concat([p for k, p in parts if k == "train"]
                      ).sample(frac=1, random_state=seed)
    val = pd.concat([p for k, p in parts if k == "val"]
                    ).sample(frac=1, random_state=seed)
    test = pd.concat([p for k, p in parts if k == "test"]
                     ).sample(frac=1, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _grouped_split(df, group_col, val_frac, test_frac, seed):
    rng = random.Random(seed)
    parts = []
    for _, g in df.groupby("target"):
        groups = list(g[group_col].dropna().astype(str).unique())
        rng.shuffle(groups)
        n = len(groups)
        n_test = int(round(test_frac * n))
        n_val = int(round(val_frac * n))
        test_g = set(groups[:n_test])
        val_g = set(groups[n_test:n_test + n_val])
        train_g = set(groups[n_test + n_val:])
        train = g[g[group_col].astype(str).isin(train_g)]
        val = g[g[group_col].astype(str).isin(val_g)]
        test = g[g[group_col].astype(str).isin(test_g)]
        parts += [("train", train), ("val", val), ("test", test)]
    train = pd.concat([p for k, p in parts if k == "train"]
                      ).sample(frac=1, random_state=seed)
    val = pd.concat([p for k, p in parts if k == "val"]
                    ).sample(frac=1, random_state=seed)
    test = pd.concat([p for k, p in parts if k == "test"]
                     ).sample(frac=1, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

# ---------- datasets ----------


class ISICSingle(Dataset):
    def __init__(self, images_dir: str, table: pd.DataFrame, train: bool):
        self.dir = images_dir
        self.tfm = _build_transforms(train)
        keep = []
        for _, r in table.iterrows():
            p = _resolve_path(images_dir, str(r["isic_id"]))
            if p:
                keep.append((p, int(r["target"])))
        self.paths = [p for p, _ in keep]
        self.labels = [y for _, y in keep]

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        x = self.tfm(img)
        y = int(self.labels[idx])
        return x, torch.tensor(y, dtype=torch.long)


class ISICTriplet(Dataset):
    """Returns (anchor, positive, negative, label_of_anchor)."""

    def __init__(self, images_dir: str, table: pd.DataFrame, train: bool):
        self.dir = images_dir
        self.tfm = _build_transforms(train)
        self.records: List[tuple[str, int]] = []
        self.class_to_indices: Dict[int, List[int]] = defaultdict(list)
        for _, r in table.iterrows():
            p = _resolve_path(images_dir, str(r["isic_id"]))
            if not p:
                continue
            y = int(r["target"])
            self.class_to_indices[y].append(len(self.records))
            self.records.append((p, y))
        assert 0 in self.class_to_indices and 1 in self.class_to_indices, "need both classes"

    def __len__(self): return len(self.records)

    def _load(self, idx: int):
        p, y = self.records[idx]
        return self.tfm(Image.open(p).convert("RGB")), y

    def __getitem__(self, idx: int):
        xa, ya = self._load(idx)
        pos_pool = self.class_to_indices[ya]
        pos_idx = idx
        if len(pos_pool) > 1:
            import random as _r
            while pos_idx == idx:
                pos_idx = _r.choice(pos_pool)
        xp, _ = self._load(pos_idx)
        yn = 1 - ya
        import random as _r
        neg_idx = _r.choice(self.class_to_indices[yn])
        xn, _ = self._load(neg_idx)
        return xa, xp, xn, torch.tensor(ya, dtype=torch.long)


class BalancedAnchorBatchSampler(BatchSampler):
    """Balanced anchors per class; with-replacement; covers dataset approx once/epoch."""

    def __init__(self, dataset: ISICTriplet, batch_size: int, seed: int = 42):
        self.dataset = dataset
        self.bs = max(2, int(batch_size))
        self.rng = random.Random(seed)
        self.c0 = dataset.class_to_indices[0][:]
        self.c1 = dataset.class_to_indices[1][:]
        self.k0 = self.bs // 2
        self.k1 = self.bs - self.k0
        # number of batches that roughly covers the minority class once
        self.n_batches = max(1, min(
            len(self.c0) // max(1, self.k0),
            len(self.c1) // max(1, self.k1)
        ))

    def __iter__(self):
        for _ in range(self.n_batches):
            b0 = (self.rng.sample(self.c0, self.k0) if len(self.c0) >= self.k0
                  else [self.rng.choice(self.c0) for _ in range(self.k0)])
            b1 = (self.rng.sample(self.c1, self.k1) if len(self.c1) >= self.k1
                  else [self.rng.choice(self.c1) for _ in range(self.k1)])
            batch = b0 + b1
            self.rng.shuffle(batch)
            yield batch

    def __len__(self): return self.n_batches

# ---------- loaders ----------


def _dl_kwargs():
    numw = int(getattr(config, "NUM_WORKERS", 0))
    kw = dict(num_workers=numw, pin_memory=True)
    if numw > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    return kw


def make_loaders():
    df = _read_metadata(config.META_CSV)

    # Filter rows to only those images that actually exist in IMAGES_DIR
    stems = set()
    for pat in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
        stems |= {os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(config.IMAGES_DIR, pat))}
    before = len(df)
    df = df[df["isic_id"].isin(stems)].reset_index(drop=True)
    print(f"[data] kept {len(df)}/{before} rows with existing files")

    if config.USE_PATIENT_SPLIT and "patient_id" in df.columns:
        train_df, val_df, test_df = _grouped_split(
            df, "patient_id", config.VAL_FRACTION, config.TEST_FRACTION, config.SEED)
    else:
        train_df, val_df, test_df = _stratified_split(
            df, config.VAL_FRACTION, config.TEST_FRACTION, config.SEED)

    train_ds = ISICSingle(config.IMAGES_DIR, train_df, train=True)
    val_ds = ISICSingle(config.IMAGES_DIR, val_df,   train=False)
    test_ds = ISICSingle(config.IMAGES_DIR, test_df,  train=False)

    dlkw = _dl_kwargs()
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,  **dlkw)
    val_loader = DataLoader(
        val_ds,   batch_size=config.BATCH_SIZE, shuffle=False, **dlkw)
    test_loader = DataLoader(
        test_ds,  batch_size=config.BATCH_SIZE, shuffle=False, **dlkw)
    return train_loader, val_loader, test_loader, (train_df, val_df, test_df)


def make_triplet_loaders_from_splits(train_df, val_df):
    train_tri = ISICTriplet(config.IMAGES_DIR, train_df, train=True)
    val_tri = ISICTriplet(config.IMAGES_DIR, val_df,   train=False)

    dlkw = _dl_kwargs()
    tr_loader = DataLoader(
        train_tri,
        batch_sampler=BalancedAnchorBatchSampler(
            train_tri, batch_size=config.BATCH_SIZE, seed=config.SEED),
        **dlkw
    )

    n0 = len(val_tri.class_to_indices[0])
    n1 = len(val_tri.class_to_indices[1])
    minority = max(1, min(n0, n1))
    val_bs = max(2, min(config.BATCH_SIZE, 2 * min(minority, 32)))
    va_loader = DataLoader(
        val_tri,
        batch_sampler=BalancedAnchorBatchSampler(
            val_tri, batch_size=val_bs, seed=config.SEED),
        **dlkw
    )
    return tr_loader, va_loader
