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


