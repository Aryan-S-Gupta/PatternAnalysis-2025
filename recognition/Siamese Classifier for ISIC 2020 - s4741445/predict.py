import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay, roc_curve
import matplotlib.pyplot as plt

import config
from data import make_loaders
from models import SiameseTriplet, HeadBinaryClassifier

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p): os.makedirs(p, exist_ok=True)


@torch.no_grad()
def denorm(x):
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = x.clone().float().cpu()
    x = x * IMAGENET_STD + IMAGENET_MEAN
    return x.clamp(0, 1).squeeze(0)


def tta_variants(x: torch.Tensor, n: int = 1):
    outs = [x]
    if n >= 2:
        outs.append(torch.flip(x, [-1]))          # hflip
    if n >= 4:
        outs.append(torch.flip(x, [-2]))                  # vflip
        outs.append(torch.transpose(x, -1, -2))           # transpose
    if n >= 8:
        outs += [torch.rot90(x, k, dims=(-2, -1)) for k in (1, -1, 2, -2)]
    return outs[:n]
