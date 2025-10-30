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


def best_accuracy_threshold(y_true_np, prob_np):
    # Dense grid for stable argmax of accuracy
    thr_grid = np.linspace(0.0, 1.0, 1001)
    accs = [((prob_np >= t).astype(int) == y_true_np).mean() for t in thr_grid]
    i = int(np.argmax(accs))
    return float(thr_grid[i]), float(accs[i])


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unpack_loader(out):
    # Supports both (P,H,Y) and (P,H,Y,Z,keep_imgs,keep_probs,keep_labels)
    if len(out) == 3:
        P, H, Y = out
        return P, H, Y, None, None, None, None
    elif len(out) == 7:
        return out
    else:
        raise ValueError(
            f"predict_loader returned unexpected arity: {len(out)}")


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


def pick_threshold_balanced(y_true_np, prob_np, min_pos_rate=0.05, max_pos_rate=0.95):
    fpr, tpr, thr = roc_curve(y_true_np, prob_np)
    # Youden's J = TPR - FPR  (maximizes balanced accuracy)
    j = tpr - fpr
    # Filter out thresholds that would predict ~all-benign or ~all-malignant
    pos_rate = (prob_np[:, None] >= thr[None, :]).mean(
        axis=0)  # predicted positive fraction
    mask = (pos_rate >= min_pos_rate) & (pos_rate <= max_pos_rate)
    if mask.any():
        idx = mask.nonzero()[0][j[mask].argmax()]
    else:
        idx = j.argmax()  # fallback
    return float(thr[idx])


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

    # --- VALIDATION: choose threshold (balanced) ---
    print("[VALIDATION] started")
    v_probs, _, v_labels = predict_loader(
        siam, clf, val_loader, device, tta=args.tta)
    t_star = pick_threshold_balanced(
        v_labels, v_probs, min_pos_rate=0.05, max_pos_rate=0.95)

    # report proper metrics
    from sklearn.metrics import roc_auc_score, confusion_matrix
    val_auc = roc_auc_score(v_labels, v_probs.astype(np.float32))
    v_pred = (v_probs >= t_star).astype(int)
    cm = confusion_matrix(v_labels, v_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / max(1, tp+fn)  # TPR
    spec = tn / max(1, tn+fp)  # TNR
    bal_acc = 0.5*(sens+spec)
    acc = (v_pred == v_labels).mean()
    print(
        f"[VALIDATION] auc_roc={val_auc:.3f} | t*={t_star:.3f} | acc={acc:.3f} | bal_acc={bal_acc:.3f} | sens={sens:.3f} | spec={spec:.3f}")
    print("[VALIDATION] finished")

    # --- TEST with the same t* ---
    print("[TEST] started")
    t_probs, _, t_labels = predict_loader(
        siam, clf, test_loader, device, tta=args.tta)
    t_pred = (t_probs >= t_star).astype(int)
    from sklearn.metrics import roc_auc_score
    test_auc = roc_auc_score(t_labels, t_probs.astype(np.float32))
    cm = confusion_matrix(t_labels, t_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / max(1, tp+fn)
    spec = tn / max(1, tn+fp)
    bal_acc = 0.5*(sens+spec)
    acc = (t_pred == t_labels).mean()
    print(
        f"[TEST] auc_roc={test_auc:.3f} | acc={acc:.3f} | bal_acc={bal_acc:.3f} | sens={sens:.3f} | spec={spec:.3f}")

    # plots at chosen threshold + ROC
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
    import matplotlib.pyplot as plt

    roc_p = os.path.join(config.ARTIFACTS, "roc_curve_test.png")
    cm_best = os.path.join(config.ARTIFACTS, "confusion_matrix_test_best.png")

    cm = confusion_matrix(t_labels, (t_probs >= t_star).astype(int))
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"]).plot(
        ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix (test, thr={t_star:.3f})")
    plt.tight_layout()
    plt.savefig(cm_best, dpi=150)
    plt.close()

    auc = roc_auc_score(t_labels, t_probs.astype(np.float32))
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(
        t_labels, t_probs, ax=ax, name=f"AUC={auc:.3f}")
    ax.set_title("ROC (test)")
    plt.tight_layout()
    plt.savefig(roc_p, dpi=150)
    plt.close()

    print("[PREDICT] saved:")
    for p in [cm_best, roc_p]:
        if os.path.exists(p):
            print("  -", p)

    print("[TEST] finished")
    print("[PREDICT] finished")


if __name__ == "__main__":
    main()
