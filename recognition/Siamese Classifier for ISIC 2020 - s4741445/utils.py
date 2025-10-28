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


@torch.no_grad()
def accuracy(clf, X, y, device):
    clf.eval()
    correct = total = 0
    for i in range(0, len(X), 512):
        xb = X[i:i+512].to(device)
        yb = y[i:i+512].to(device)
        pred = clf(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / max(1, total)


def plot_tsne(X: torch.Tensor, y: torch.Tensor, out_path: str):
    Xn = X.float().cpu().numpy()
    yn = y.cpu().numpy()
    Z = TSNE(n_components=2, perplexity=30, learning_rate="auto",
             init="pca", random_state=0).fit_transform(Xn)
    plt.figure()
    plt.scatter(Z[yn == 0, 0], Z[yn == 0, 1], s=6, alpha=0.7, label="Benign")
    plt.scatter(Z[yn == 1, 0], Z[yn == 1, 1],
                s=6, alpha=0.7, label="Malignant")
    plt.legend()
    plt.title("t-SNE (train embeddings)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (test)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_roc_curve(y_true, y_score, out_path):
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(
        y_true, y_score, ax=ax, name=f"AUC={auc:.3f}")
    ax.set_title("ROC (test)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def save_distance_hist(X: torch.Tensor, y: torch.Tensor, out_path: str):
    X = torch.nn.functional.normalize(X.float(), dim=1)
    D = torch.cdist(X, X, p=2.0).cpu().numpy()
    y = y.cpu().numpy()
    pos = []
    neg = []
    for i in range(len(y)):
        same = (y == y[i])
        pos.extend(D[i][same & (np.arange(len(y)) != i)])
        neg.extend(D[i][~same])
    plt.figure()
    plt.hist(pos, bins=50, alpha=0.6, label="pos")
    plt.hist(neg, bins=50, alpha=0.6, label="neg")
    plt.legend()
    plt.title("Pairwise distance distribution (val)")
    plt.xlabel("L2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def save_three_panel(xs, y_left, y_mid, y_right,
                     labels_left, labels_mid, labels_right,
                     title_left, title_mid, title_right,
                     out_path, smooth_k=1):
    plt.figure(figsize=(18, 5))
    ax1 = plt.subplot(1, 3, 1)
    for y, l in zip(y_left, labels_left):
        ax1.plot(xs, moving_avg(y, smooth_k), label=l)
    ax1.set_title(title_left)
    ax1.set_xlabel("epoch")
    ax1.legend()

    ax2 = plt.subplot(1, 3, 2)
    for y, l in zip(y_mid, labels_mid):
        ax2.plot(xs, moving_avg(y, smooth_k), label=l)
    ax2.set_title(title_mid)
    ax2.set_xlabel("epoch")
    ax2.legend()

    ax3 = plt.subplot(1, 3, 3)
    for y, l in zip(y_right, labels_right):
        ax3.plot(xs, moving_avg(y, smooth_k), label=l)
    ax3.set_title(title_right)
    ax3.set_xlabel("epoch")
    ax3.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
