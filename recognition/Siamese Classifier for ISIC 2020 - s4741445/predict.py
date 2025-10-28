import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay, roc_curve
)
import matplotlib.pyplot as plt

import config
from data import make_loaders
from modules import SiameseTriplet, HeadBinaryClassifier

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path): os.makedirs(path, exist_ok=True)


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
        outs.append(torch.flip(x, [-1]))                 # hflip
    if n >= 4:
        outs.append(torch.flip(x, [-2]))                 # vflip
        outs.append(torch.transpose(x, -1, -2))          # transpose
    if n >= 8:
        outs += [torch.rot90(x, k, dims=(-2, -1)) for k in (1, -1, 2, -2)]
    return outs[:n]


def try_load_state(model, path, strict=True):
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict(sd, strict=strict)


def load_models(device, siam_path="", clf_path=""):
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
        # Fallback to aux head EMA if classifier.pt not present
        try_load_state(clf, ema_aux, strict=False)
        print("[WARN] classifier.pt not found; using aux_head_ema.pt fallback.")
    else:
        raise FileNotFoundError(
            "No classifier weights found. Provide --classifier or train first.")

    siam.eval()
    clf.eval()
    return siam, clf


@torch.no_grad()
def predict_loader(siam, clf, loader: DataLoader, device, tta: int = 1):
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


def plot_confmat(y_true, y_pred, out_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"])
    fig, ax = plt.subplots()
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc(y_true, y_score, out_path):
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(
        y_true, y_score, ax=ax, name=f"AUC={auc:.3f}")
    ax.set_title("ROC (test)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
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

    # data (CSV + on-disk image presence handled in make_loaders)
    _, val_loader, test_loader, _ = make_loaders()

    # models
    siam, clf = load_models(device, args.siamese, args.classifier)

    # --- VALIDATION: choose threshold ---
    print("[VALIDATION] started")
    v_probs, _, v_labels = predict_loader(
        siam, clf, val_loader, device, tta=args.tta)
    val_auc = roc_auc_score(v_labels, v_probs)
    fpr, tpr, thr = roc_curve(v_labels, v_probs)
    j = tpr - fpr
    t_star = thr[np.argmax(j)]
    val_acc_05 = ((v_probs >= 0.5).astype(int) == v_labels).mean()
    val_acc_ts = ((v_probs >= t_star).astype(int) == v_labels).mean()
    print(
        f"[VALIDATION] auc_roc={val_auc:.3f} | accuracy@0.5={val_acc_05:.3f} | accuracy@t*={val_acc_ts:.3f} | t*={t_star:.3f}")
    print("[VALIDATION] finished")

    # --- TEST: evaluate ---
    print("[TEST] started")
    t_probs, _, t_labels = predict_loader(
        siam, clf, test_loader, device, tta=args.tta)
    test_auc = roc_auc_score(t_labels, t_probs)
    preds_05 = (t_probs >= 0.5).astype(int)
    preds_ts = (t_probs >= t_star).astype(int)
    acc_05 = (preds_05 == t_labels).mean()
    acc_ts = (preds_ts == t_labels).mean()
    print(
        f"[TEST] auc_roc={test_auc:.3f} | accuracy@0.5={acc_05:.3f} | accuracy@t*={acc_ts:.3f}")

    # plots (only what you want)
    cm05 = os.path.join(config.ARTIFACTS, "confusion_matrix_test_05.png")
    cmts = os.path.join(config.ARTIFACTS, "confusion_matrix_test_tstar.png")
    roc_p = os.path.join(config.ARTIFACTS, "roc_curve_test.png")

    plot_confmat(t_labels, preds_05, cm05,
                 title="Confusion Matrix (test, thr=0.5)")
    plot_confmat(t_labels, preds_ts, cmts,
                 title=f"Confusion Matrix (test, thr={t_star:.2f})")
    plot_roc(t_labels, t_probs, roc_p)

    print("[PREDICT] saved:")
    for p in [cm05, cmts, roc_p]:
        if os.path.exists(p):
            print("  -", p)

    print("[TEST] finished")
    print("[PREDICT] finished")


if __name__ == "__main__":
    main()
