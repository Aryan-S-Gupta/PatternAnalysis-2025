import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, RocCurveDisplay, roc_auc_score, ConfusionMatrixDisplay


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def moving_avg(x, k=3):
    if k <= 1 or len(x) < 2:
        return x
    k = min(k, len(x))
    w = np.ones(k)/k
    return np.convolve(np.array(x, dtype=float), w, mode="same").tolist()


def plot_curves(train, val, title, out_dir, fname, smooth_k=1):
    os.makedirs(out_dir, exist_ok=True)
    t = moving_avg(train, smooth_k)
    v = moving_avg(val,   smooth_k)
    plt.figure()
    plt.title(title)
    plt.plot(t, label="train")
    plt.plot(v, label="val")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    p = os.path.join(out_dir, fname)
    plt.savefig(p, dpi=140)
    plt.close()
    return p

