"""
Prediction and evaluation script for the ISIC 2020 pipeline.
Loads trained weights, runs TTA, prints metrics, and saves ROC/CM plots.
Made by Aryan Somesh Gupta (s47414451)
"""

import os
import argparse
import numpy as np
import torch
import config
from data import make_loaders
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from modules import SiameseTriplet, HeadBinaryClassifier
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay, roc_curve
)

# Constants for normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Set random seeds for reproducibility


def set_seed(seed=42):
    """Seed Python, NumPy, and Torch for reproducible inference."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Ensure directory exists


def ensure_dir(path):
    """Create the directory `path` if it does not exist (no-op otherwise)."""
    os.makedirs(path, exist_ok=True)

# Denormalize image tensor


@torch.no_grad()
def denorm(x):
    """Undo ImageNet normalization on a BCHW/CHW tensor and clamp to [0,1]."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = x.clone().float().cpu()
    x = x * IMAGENET_STD + IMAGENET_MEAN
    return x.clamp(0, 1).squeeze(0)

# Generate TTA variants


def tta_variants(x: torch.Tensor, n: int = 1):
    """Generate up to {1,2,4,8} deterministic TTA variants (flip/transpose/rot90)."""
    outs = [x]
    if n >= 2:
        outs.append(torch.flip(x, [-1]))         # hflip
    if n >= 4:
        outs.append(torch.flip(x, [-2]))                 # vflip
        outs.append(torch.transpose(x, -1, -2))          # transpose
    if n >= 8:
        outs += [torch.rot90(x, k, dims=(-2, -1)) for k in (1, -1, 2, -2)]
    return outs[:n]

# Load model state dict with error handling


def try_load_state(model, path, strict=True):
    """Load a state_dict from `path` into `model` with optional `strict` flag."""
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict(sd, strict=strict)

# Load Siamese and Classifier models with weights


def load_models(device, siam_path="", clf_path=""):
    """Instantiate Siamese + classifier on `device` and load weights.
    Prefers explicit paths, then EMA/final fallbacks under `config.ARTIFACTS`.
    """
    siam = SiameseTriplet().to(device)
    clf = HeadBinaryClassifier().to(device)

    ema_siam = os.path.join(config.ARTIFACTS, "siamese_stage1_ema.pt")
    final_s = os.path.join(config.ARTIFACTS, "siamese_final.pt")
    final_clf = os.path.join(config.ARTIFACTS, "classifier.pt")
    ema_aux = os.path.join(config.ARTIFACTS, "aux_head_ema.pt")

    if siam_path and os.path.exists(siam_path):
        try_load_state(siam, siam_path)
    elif os.path.exists(ema_siam):
        try_load_state(siam, ema_siam)
    elif os.path.exists(final_s):
        try_load_state(siam, final_s)
    else:
        raise FileNotFoundError(
            "No Siamese weights found. Provide --siamese or train first.")

    if clf_path and os.path.exists(clf_path):
        try_load_state(clf, clf_path)
    elif os.path.exists(final_clf):
        try_load_state(clf, final_clf)
    elif os.path.exists(ema_aux):
        try_load_state(clf, ema_aux, strict=False)
        print("[WARN] classifier.pt not found; using aux_head_ema.pt fallback.")
    else:
        raise FileNotFoundError(
            "No classifier weights found. Provide --classifier or train first.")

    siam.eval()
    clf.eval()
    return siam, clf

# Predict over a DataLoader with optional TTA


@torch.no_grad()
def predict_loader(siam, clf, loader: DataLoader, device, tta: int = 1):
    """Run a full loader, optionally with TTA, returning `(probs, preds, labels)`."""
    probs_all, preds_all, labels_all = [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if tta <= 1:
            z = siam.forward_once(xb)
            logits = clf(z)
        else:
            acc_logits = None
            for xa in tta_variants(xb, tta):
                za = siam.forward_once(xa)
                la = clf(za)
                acc_logits = la if acc_logits is None else (acc_logits + la)
            logits = acc_logits / float(tta)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(1)
        probs_all.append(probs.float().cpu())
        preds_all.append(preds.cpu())
        labels_all.append(yb.cpu())
    P = torch.cat(probs_all).numpy()
    Y = torch.cat(labels_all).numpy()
    H = torch.cat(preds_all).numpy()
    return P, H, Y

# Confusion matrix plot


def plot_confmat(y_true, y_pred, out_path, title="Confusion Matrix"):
    """Save a labeled confusion matrix plot to `out_path`."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"])
    fig, ax = plt.subplots()
    disp.plot(ax=ax, colorbar=True)  # show scale
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ROC curve plot


def plot_roc(y_true, y_score, out_path, title="ROC (test)"):
    """Save a ROC curve with AUC in the legend to `out_path`."""
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(
        y_true, y_score, ax=ax, name=f"AUC={auc:.3f}")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# Compute metrics from predictions


def metrics_from_preds(y_true, y_pred):
    """Compute accuracy, balanced accuracy, sensitivity, and specificity."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / max(1, tp+fn)  # recall positive
    spec = tn / max(1, tn+fp)  # recall negative
    bal_acc = 0.5*(sens+spec)
    acc = (y_pred == y_true).mean()
    return acc, bal_acc, sens, spec

# Choose threshold that balances accuracy and balanced accuracy


def pick_threshold_compromise(y, p, steps=1001, min_sens=0.70):
    """Grid-search a probability threshold that balances acc & bal-acc while enforcing a minimum sensitivity."""
    y = np.asarray(y)
    p = np.asarray(p)
    ts = np.linspace(0.0, 1.0, steps)
    preds = (p[:, None] >= ts).astype(int)
    acc = (preds == y[:, None]).mean(axis=0)

    sens = np.zeros_like(ts)
    spec = np.zeros_like(ts)
    bal = np.zeros_like(ts)
    for i in range(len(ts)):
        cm = confusion_matrix(y, preds[:, i], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        s = tp / max(1, tp+fn)
        c = tn / max(1, tn+fp)
        sens[i] = s
        spec[i] = c
        bal[i] = 0.5*(s+c)

    score = 0.5*acc + 0.5*bal
    score[sens < min_sens] = -1e9  # enforce min sensitivity
    i = int(score.argmax())
    return float(ts[i]), dict(acc=float(acc[i]), bal=float(bal[i]),
                              sens=float(sens[i]), spec=float(spec[i]))

# Main prediction routine


def main():
    """CLI entrypoint: load data/models, tune threshold on val, evaluate on test, and write plots & summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--siamese", default="", type=str,
                        help="Path to Siamese weights")
    parser.add_argument("--classifier", default="", type=str,
                        help="Path to classifier weights")
    parser.add_argument("--tta", default=1, type=int,
                        choices=[1, 2, 4, 8], help="Test-time augmentation views")
    args = parser.parse_args()

    set_seed(config.SEED)
    ensure_dir(config.ARTIFACTS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[PREDICT] started")

    #  Data
    _, val_loader, test_loader, _ = make_loaders()

    #  Models
    siam, clf = load_models(device, args.siamese, args.classifier)

    #  VALIDATION: choose single compromise threshold
    print("[VALIDATION] started")
    v_probs, _, v_labels = predict_loader(
        siam, clf, val_loader, device, tta=args.tta)
    val_auc = roc_auc_score(v_labels, v_probs.astype(np.float32))
    t_star, stats = pick_threshold_compromise(
        v_labels, v_probs, steps=1001, min_sens=0.70)

    print(f"[VALIDATION] AUC={val_auc:.3f} | t*={t_star:.3f} | "
          f"acc={stats['acc']:.3f} | bal_acc={stats['bal']:.3f} | "
          f"sens={stats['sens']:.3f} | spec={stats['spec']:.3f}")
    print("[VALIDATION] finished")

    #  TEST with that same threshold
    print("[TEST] started")
    t_probs, _, t_labels = predict_loader(
        siam, clf, test_loader, device, tta=args.tta)
    test_auc = roc_auc_score(t_labels, t_probs.astype(np.float32))
    t_pred = (t_probs >= t_star).astype(int)
    acc, bal_acc, sens, spec = metrics_from_preds(t_labels, t_pred)

    #  Concise final summary
    print("\n================= FINAL TEST RESULTS =================")
    print(f"Overall Accuracy      : {acc * 100:.2f}%")
    print(f"Area Under Curve (AUC): {test_auc:.3f}")
    print(f"Balanced Accuracy     : {bal_acc * 100:.2f}%")
    print(f"Sensitivity (Recall+) : {sens * 100:.2f}%")
    print(f"Specificity (Recall−) : {spec * 100:.2f}%")
    print("======================================================\n")

    #  Plots
    cm_path = os.path.join(config.ARTIFACTS, "confusion_matrix_test.png")
    roc_path = os.path.join(config.ARTIFACTS, "roc_curve_test.png")
    plot_confmat(t_labels, t_pred, cm_path,
                 title=f"Confusion Matrix (test, thr={t_star:.3f})")
    plot_roc(t_labels, t_probs, roc_path)

    print("[PREDICT] saved:")
    for p in [cm_path, roc_path]:
        if os.path.exists(p):
            print(f"  - {p}")

    print("[TEST] finished")
    print("[PREDICT] finished")


if __name__ == "__main__":
    main()
