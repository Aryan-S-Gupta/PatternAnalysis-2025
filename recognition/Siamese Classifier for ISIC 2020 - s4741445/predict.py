# predict.py

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
    probs_all, preds_all, labels_all, embeds_all = [], [], [], []
    keep_imgs, keep_probs, keep_labels = [], [], []

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
            z = siam.forward_once(xb)  # canonical emb for diagnostics

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(1)

        probs_all.append(probs.float().cpu())
        preds_all.append(preds.cpu())
        labels_all.append(yb.cpu())
        embeds_all.append(z.float().cpu())

        # cache a few for grid
        k = min(4, xb.size(0))
        keep_imgs += [xb[i].detach().cpu() for i in range(k)]
        keep_probs += [probs[i].detach().cpu() for i in range(k)]
        keep_labels += [yb[i].detach().cpu() for i in range(k)]

    P = torch.cat(probs_all).numpy()
    Y = torch.cat(labels_all).numpy()
    H = torch.cat(preds_all).numpy()
    Z = torch.cat(embeds_all, 0)

    keep_probs = torch.stack(keep_probs) if keep_probs else torch.empty(0)
    keep_labels = torch.stack(keep_labels) if keep_labels else torch.empty(0)
    return P, H, Y, Z, keep_imgs, keep_probs, keep_labels


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


def plot_tsne(Z: torch.Tensor, y, out_path):
    from sklearn.manifold import TSNE
    Xn = Z.float().numpy()
    yn = np.asarray(y)
    Z2 = TSNE(n_components=2, perplexity=30, learning_rate="auto",
              init="pca", random_state=0).fit_transform(Xn)
    plt.figure()
    plt.scatter(Z2[yn == 0, 0], Z2[yn == 0, 1], s=6, alpha=0.7, label="Benign")
    plt.scatter(Z2[yn == 1, 0], Z2[yn == 1, 1],
                s=6, alpha=0.7, label="Malignant")
    plt.legend()
    plt.title("t-SNE (test embeddings)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_sample_grid(imgs, probs, labels, out_path, k=25):
    if len(imgs) == 0 or probs.numel() == 0:
        return
    k = min(k, len(imgs))
    p = probs.float().numpy()
    l = labels.numpy()
    idx = np.argsort(np.abs(p - 0.5))[:k]
    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 1.8 * rows))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i >= k:
            continue
        img = denorm(imgs[idx[i]])
        ax.imshow(img.permute(1, 2, 0).numpy())
        pred = int(p[idx[i]] > 0.5)
        true = int(l[idx[i]])
        ok = (pred == true)
        ax.set_title(f"{'M' if pred else 'B'} {p[idx[i]]:.2f}",
                     color=("green" if ok else "red"), fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
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
    v_probs, _, v_labels, _, _, _, _ = predict_loader(
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
    t_probs, _, t_labels, Z, keep_imgs, keep_probs, keep_labels = predict_loader(
        siam, clf, test_loader, device, tta=args.tta
    )
    test_auc = roc_auc_score(t_labels, t_probs)
    preds_05 = (t_probs >= 0.5).astype(int)
    preds_ts = (t_probs >= t_star).astype(int)
    acc_05 = (preds_05 == t_labels).mean()
    acc_ts = (preds_ts == t_labels).mean()
    print(
        f"[TEST] auc_roc={test_auc:.3f} | accuracy@0.5={acc_05:.3f} | accuracy@t*={acc_ts:.3f}")

    # plots
    cm05 = os.path.join(config.ARTIFACTS, "confusion_matrix_test_05.png")
    cmts = os.path.join(config.ARTIFACTS, "confusion_matrix_test_tstar.png")
    roc = os.path.join(config.ARTIFACTS, "roc_curve_test.png")
    tsne = os.path.join(config.ARTIFACTS, "tsne_test.png")
    grid = os.path.join(config.ARTIFACTS, "sample_predictions.png")

    plot_confmat(t_labels, preds_05, cm05,
                 title="Confusion Matrix (test, thr=0.5)")
    plot_confmat(t_labels, preds_ts, cmts,
                 title=f"Confusion Matrix (test, thr={t_star:.2f})")
    plot_roc(t_labels, t_probs, roc)
    try:
        plot_tsne(Z, t_labels, tsne)
    except Exception as e:
        print(f"[PREDICT] t-SNE skipped: {e}")
    try:
        plot_sample_grid(keep_imgs, keep_probs, keep_labels, grid, k=25)
    except Exception as e:
        print(f"[PREDICT] sample grid skipped: {e}")

    print("[PREDICT] saved:")
    for p in [cm05, cmts, roc, tsne, grid]:
        if os.path.exists(p):
            print("  -", p)

    print("[TEST] finished")
    print("[PREDICT] finished")


if __name__ == "__main__":
    main()
