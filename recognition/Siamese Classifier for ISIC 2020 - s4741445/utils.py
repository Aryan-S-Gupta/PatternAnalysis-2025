import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, RocCurveDisplay, roc_auc_score, ConfusionMatrixDisplay

# set random seeds
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# moving average for smoothing plots
def moving_avg(x, k=3):
    if k <= 1 or len(x) < 2:
        return x
    k = min(k, len(x))
    w = np.ones(k) / k
    return np.convolve(np.array(x, dtype=float), w, mode="same").tolist()

# plot training vs validation curves
def plot_curves(train, val, title, out_dir, fname, smooth_k=1):
    os.makedirs(out_dir, exist_ok=True)
    t = moving_avg(train, smooth_k)
    v = moving_avg(val, smooth_k)
    plt.figure()
    plt.title(title)
    plt.plot(t, label="train")
    plt.plot(v, label="val")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=140)
    plt.close()
    return path

# plot a single curve
def plot_curve(series, title, out_dir, fname, smooth_k=1, ylabel="value"):
    os.makedirs(out_dir, exist_ok=True)
    y = moving_avg(series, smooth_k)
    plt.figure()
    plt.title(title)
    plt.plot(y, label="value")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=140)
    plt.close()
    return path

# compute accuracy
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

# confusion matrix plot
def plot_confusion_matrix(y_true, y_pred, labels, out_path):
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, colorbar=True)
    ax.set_title("Confusion Matrix (test)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()

# ROC curve plot
def plot_roc_curve(y_true, y_score, out_path):
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    else:
        y_true = np.asarray(y_true)
    if torch.is_tensor(y_score):
        y_score = y_score.detach().cpu().to(torch.float32).numpy()
    else:
        y_score = np.asarray(y_score, dtype=np.float32)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(
        y_true, y_score, ax=ax, name=f"AUC={auc:.3f}")
    ax.set_title("ROC (test)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
