import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from torchvision.transforms import v2
from sklearn.decomposition import PCA
from torch.utils.data import Subset
import config
from data import make_loaders, make_triplet_loaders_from_splits
from modules import SiameseTriplet, HeadBinaryClassifier
from utils import (
    set_seed, plot_curves, accuracy, plot_confusion_matrix,
    plot_roc_curve, plot_curve
)

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")  # TF32 on Ampere+
except Exception:
    pass
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# --- helpers (local) ---

def d_cos_pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - (a * b).sum(dim=1).clamp(-1, 1)


def supcon_loss(z: torch.Tensor, y: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    z = nn.functional.normalize(z, dim=1)
    sim = z @ z.t() / tau
    sim = sim - torch.eye(sim.size(0), device=sim.device) * 1e9
    y = y.view(-1, 1)
    pos_mask = (y == y.t()).float()
    pos_mask.fill_diagonal_(0.0)
    denom = torch.logsumexp(sim, dim=1, keepdim=True)
    log_prob = sim - denom
    pos_count = pos_mask.sum(1).clamp(min=1.0)
    loss = -(pos_mask * log_prob).sum(1) / pos_count
    return loss.mean()


class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            sv = self.shadow.get(k, None)
            if sv is None:
                self.shadow[k] = v.detach().clone()
                continue
            if v.is_floating_point():
                sv.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                sv.copy_(v)

    def load_into(self, model):
        model.load_state_dict(self.shadow, strict=True)


def pick_threshold_max_accuracy(y, p, steps=400):
    y = np.asarray(y)
    p = np.asarray(p)
    ts = np.linspace(0.0, 1.0, steps)
    accs = [(((p >= t).astype(int) == y).mean(), t) for t in ts]
    acc, t_star = max(accs, key=lambda x: x[0])
    return float(t_star), float(acc)


@torch.no_grad()
def cache_embeddings(siam: SiameseTriplet, loader: DataLoader, device, autocast_kwargs):
    X, y = [], []
    siam.eval()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).to(
            memory_format=torch.channels_last)
        with torch.autocast(**autocast_kwargs):
            z = siam.forward_once(xb)
        X.append(z.cpu())
        y.append(yb)
    return torch.cat(X, 0), torch.cat(y, 0)

# --- NEW: viz helpers (self-contained; no changes needed in utils.py) ---


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])[None, None, :]
_IMAGENET_STD = np.array([0.229, 0.224, 0.225])[None, None, :]


def _denorm_batch(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().permute(0, 2, 3, 1).numpy()
    x = x * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(x, 0, 1)


def plot_image_grid(xb: torch.Tensor, labels: torch.Tensor, out_path: str, title: str, cols: int = 8):
    imgs = _denorm_batch(xb)
    n = imgs.shape[0]
    cols = min(cols, n)
    rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = np.atleast_2d(axes)
    for i in range(rows*cols):
        ax = axes[i//cols, i % cols]
        ax.axis("off")
        if i < n:
            ax.imshow(imgs[i])
            if labels is not None:
                y = int(labels[i]) if torch.is_tensor(
                    labels) else int(labels[i])
                ax.set_title(f"y={y}", fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


# map ints → human labels
_LABELS = {0: "Benign", 1: "Malignant"}


def _to_numpy_for_display(x: torch.Tensor, denorm: bool) -> np.ndarray:
    arr = x.detach().cpu().permute(0, 2, 3, 1).numpy()
    if denorm:
        arr = arr * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(arr, 0, 1)


def plot_image_grid_with_names(xb: torch.Tensor, labels: torch.Tensor, out_path: str,
                               title: str, cols: int = 8, denorm: bool = True):
    imgs = _to_numpy_for_display(xb, denorm=denorm)
    n = imgs.shape[0]
    cols = min(cols, n)
    rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.2, rows*2.2))
    axes = np.atleast_2d(axes)
    for i in range(rows*cols):
        ax = axes[i//cols, i % cols]
        ax.axis("off")
        if i < n:
            ax.imshow(imgs[i])
            y = int(labels[i]) if torch.is_tensor(labels) else int(labels[i])
            ax.set_title(_LABELS.get(y, str(y)), fontsize=9)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_prediction_grid(xb: torch.Tensor, yb: torch.Tensor, probs: torch.Tensor, preds: torch.Tensor,
                         out_path: str, title: str = "Predictions", cols: int = 8):
    imgs = _denorm_batch(xb)
    n = imgs.shape[0]
    cols = min(cols, n)
    rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.2, rows*2.2))
    axes = np.atleast_2d(axes)
    for i in range(rows*cols):
        ax = axes[i//cols, i % cols]
        ax.axis("off")
        if i < n:
            ax.imshow(imgs[i])
            y = int(yb[i])
            pr = int(preds[i])
            p = float(probs[i])
            ok = (y == pr)
            ax.set_title(f"{_LABELS.get(y, y)} → {_LABELS.get(pr, pr)}  p1={p:.2f}",
                         fontsize=8, color=("green" if ok else "red"))
            for spine in ax.spines.values():
                spine.set_edgecolor("green" if ok else "red")
                spine.set_linewidth(2.0)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_feature_scatter_2d(X: torch.Tensor, y: torch.Tensor, out_path: str,
                            title: str = "Feature scatter (t-SNE)",
                            max_points: int = 2000, method: str = "tsne"):
    Xn = X.detach().cpu().numpy()
    yn = y.detach().cpu().numpy()

    # stratified subsample for speed
    if len(Xn) > max_points:
        keep_idx = []
        for cls in np.unique(yn):
            idx = np.where(yn == cls)[0]
            k = max(1, int(max_points * (len(idx) / len(Xn))))
            keep_idx.append(np.random.choice(idx, size=k, replace=False))
        keep_idx = np.concatenate(keep_idx)
        Xn, yn = Xn[keep_idx], yn[keep_idx]

    # speed up with PCA pre-reduction
    X50 = PCA(n_components=min(50, Xn.shape[1])).fit_transform(Xn)

    if method == "pca":
        Z = PCA(n_components=2).fit_transform(Xn)  # fastest
    else:
        tsne = TSNE(n_components=2, init="pca", learning_rate="auto",
                    perplexity=30, max_iter=750, random_state=42)
        Z = tsne.fit_transform(X50)

    plt.figure(figsize=(6, 5))
    for cls, name in [(0, "Benign"), (1, "Malignant")]:
        m = (yn == cls)
        plt.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.65, label=name)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def train():
    set_seed(config.SEED)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    autocast_kwargs = dict(
        device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda)

    os.makedirs(config.ARTIFACTS, exist_ok=True)

    # ---- loaders ----
    base_tr, base_va, base_te, (train_df, val_df, test_df) = make_loaders()
    tri_tr, tri_va = make_triplet_loaders_from_splits(train_df, val_df)

    print(
        f"[DATA] sizes: train={len(base_tr.dataset)} val={len(base_va.dataset)} test={len(base_te.dataset)}", flush=True)
    xb, yb = next(iter(base_tr))
    print(
        f"[DATA] warmup batch: {tuple(xb.shape)}, labels sample={yb[:4].tolist()}", flush=True)

    # --- NEW: sample image grids before/after augmentation ---
    # eval-style transform (no aug)
    # --- NORMALIZATION COMPARISON GRIDS (before vs after) ---
    no_norm_tfm = v2.Compose([
        v2.ToImage(),
        v2.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])
    norm_tfm = v2.Compose([
        v2.ToImage(),
        v2.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    N = min(24, len(base_tr.dataset.paths))
    idxs = random.sample(range(len(base_tr.dataset.paths)), k=N)
    x_before, x_after, y_batch = [], [], []
    for i in idxs:
        p = base_tr.dataset.paths[i]
        y = base_tr.dataset.labels[i]
        img = Image.open(p).convert("RGB")
        x_before.append(no_norm_tfm(img))  # BEFORE normalization
        x_after.append(norm_tfm(img))      # AFTER normalization
        y_batch.append(y)
    xb_before = torch.stack(x_before, 0)
    xb_after = torch.stack(x_after,  0)
    yb_batch = torch.tensor(y_batch, dtype=torch.long)

    plot_image_grid_with_names(
        xb_before, yb_batch,
        os.path.join(config.ARTIFACTS, "grid_before_normalization.png"),
        "Before normalization (Benign/Malignant labels)", cols=8, denorm=False
    )
    plot_image_grid_with_names(
        xb_after, yb_batch,
        os.path.join(config.ARTIFACTS, "grid_after_normalization.png"),
        "After normalization (Benign/Malignant labels)", cols=8, denorm=False
    )

    # ---- models ----
    siam = SiameseTriplet().to(device).to(memory_format=torch.channels_last)
    clf_aux = HeadBinaryClassifier().to(device)

    # warmup freeze (S1)
    for p in siam.embed.backbone.parameters():
        p.requires_grad = False

    optim = Adam([
        {"params": siam.embed.proj.parameters(), "lr": config.LR_SIAMESE * 5.0},
        {"params": clf_aux.parameters(),         "lr": config.LR_SIAMESE * 5.0},
    ], betas=config.BETAS, weight_decay=config.WEIGHT_DECAY)

    sched = CosineAnnealingLR(
        optim, T_max=config.EPOCHS_SIAMESE, eta_min=config.LR_SIAMESE*0.1)
    ce_aux = nn.CrossEntropyLoss(label_smoothing=0.05).to(device)

    ema_siam = EMA(siam, decay=config.EMA_DECAY)
    ema_aux = EMA(clf_aux, decay=config.EMA_DECAY)

    # schedules
    margin_start = float(getattr(config, "MARGIN_START", 0.3))
    margin_end = float(getattr(config, "MARGIN_END",   max(
        0.5, float(getattr(config, "MARGIN", 0.5)))))
    aux_lambda = float(getattr(config, "AUX_LAMBDA", 0.5))
    supcon_tau = float(getattr(config, "SUPCON_TAU", 0.07))
    supcon_warm = int(getattr(config, "SUPCON_WARMUP_EPOCHS", 0))
    smooth_k = int(getattr(config, "SMOOTH_K", 1))

    # logs
    s1_tr_loss, s1_va_loss = [], []
    s1_tr_acc,  s1_va_acc = [], []
    s1_tr_auc,  s1_va_auc = [], []
    s1_va_pos,  s1_va_neg = [], []

    best_val, best_ep = float("inf"), 0
    LOG_EVERY = 10
    VAL_PEEK_STEPS = 4

    print("[TRAIN:S1] started")

    for ep in range(1, config.EPOCHS_SIAMESE + 1):
        # unfreeze backbone
        if ep == config.FREEZE_BACKBONE_EPOCHS + 1:
            for p in siam.embed.backbone.parameters():
                p.requires_grad = True
            optim = Adam([
                {"params": siam.embed.backbone.parameters(
                ), "lr": config.LR_SIAMESE * 0.5},
                {"params": siam.embed.proj.parameters(
                ),     "lr": config.LR_SIAMESE * 1.0},
                {"params": clf_aux.parameters()},
            ], lr=config.LR_SIAMESE, betas=config.BETAS, weight_decay=config.WEIGHT_DECAY)
            remaining = max(1, config.EPOCHS_SIAMESE - ep + 1)
            sched = CosineAnnealingLR(
                optim, T_max=remaining, eta_min=config.LR_SIAMESE*0.1)

        # schedule margin
        t = (ep - 1) / max(1, (config.EPOCHS_SIAMESE - 1))
        margin = margin_start + t * (margin_end - margin_start)

        # ---- train step ----
        siam.train()
        clf_aux.train()
        tl = []
        probs = []
        ys = []
        running = {"loss": 0.0, "n": 0}

        for b_idx, (xa, xp, xn, y) in enumerate(tri_tr, start=1):
            xa = xa.to(device, non_blocking=True).to(
                memory_format=torch.channels_last)
            xp = xp.to(device, non_blocking=True).to(
                memory_format=torch.channels_last)
            xn = xn.to(device, non_blocking=True).to(
                memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            with torch.autocast(**autocast_kwargs):
                za = siam.forward_once(xa)
                zp = siam.forward_once(xp)
                zn = siam.forward_once(xn)

                if ep <= supcon_warm:
                    z_cat = torch.cat([za, zp], dim=0)
                    y_cat = torch.cat([y,  y], dim=0)
                    loss = supcon_loss(z_cat, y_cat, tau=supcon_tau)
                else:
                    d_ap = d_cos_pair(za, zp)
                    d_an = d_cos_pair(za, zn)
                    trip = torch.relu(d_ap - d_an + margin).mean()
                    logits = clf_aux(za)
                    ce = ce_aux(logits, y)
                    loss = trip + aux_lambda * ce

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(siam.parameters()) + list(clf_aux.parameters()), config.MAX_NORM)
            optim.step()

            ema_siam.update(siam)
            ema_aux.update(clf_aux)

            tl.append(loss.item())
            running["loss"] += float(loss.item())
            running["n"] += 1
            if running["n"] % LOG_EVERY == 0:
                avg = running["loss"] / running["n"]
                try:
                    total_batches = len(tri_tr)
                except Exception:
                    total_batches = running["n"]
                print(
                    f"[S1][train] ep {ep}/{config.EPOCHS_SIAMESE} b {running['n']}/{total_batches} | loss {avg:.4f}", flush=True)

            if ep > supcon_warm:
                with torch.no_grad():
                    probs.append(torch.softmax(logits, dim=1)
                                 [:, 1].float().cpu())
                    ys.append(y.cpu())

        tr_loss = float(np.mean(tl))
        s1_tr_loss.append(tr_loss)

        if ep <= supcon_warm:
            s1_tr_acc.append(np.nan)
            s1_tr_auc.append(np.nan)
        else:
            tr_p = torch.cat(probs).numpy()
            tr_y = torch.cat(ys).numpy()
            s1_tr_acc.append(((tr_p > 0.5).astype(int) == tr_y).mean())
            try:
                s1_tr_auc.append(roc_auc_score(tr_y, tr_p))
            except Exception:
                s1_tr_auc.append(0.5)

        # --- mid-epoch validation peek (few batches, EMA) ---
        siam_eval = SiameseTriplet().to(device).to(memory_format=torch.channels_last)
        ema_siam.load_into(siam_eval)
        clf_eval = HeadBinaryClassifier().to(device)
        ema_aux.load_into(clf_eval)
        siam_eval.eval()
        clf_eval.eval()
        peek_losses = []
        with torch.no_grad():
            for i, (xa, xp, xn, y) in enumerate(tri_va):
                if i >= VAL_PEEK_STEPS:
                    break
                xa = xa.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last)
                xp = xp.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last)
                xn = xn.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True)
                with torch.autocast(**autocast_kwargs):
                    za = siam_eval.forward_once(xa)
                    zp = siam_eval.forward_once(xp)
                    zn = siam_eval.forward_once(xn)
                    if ep <= supcon_warm:
                        z_cat = torch.cat([za, zp], dim=0)
                        y_cat = torch.cat([y, y], dim=0)
                        loss = supcon_loss(z_cat, y_cat, tau=supcon_tau)
                    else:
                        d_ap = d_cos_pair(za, zp)
                        d_an = d_cos_pair(za, zn)
                        trip = torch.relu(d_ap - d_an + margin).mean()
                        logits = clf_eval(za)
                        ce = ce_aux(logits, y)
                        loss = trip + aux_lambda * ce
                peek_losses.append(loss.item())
        if peek_losses:
            print(
                f"[S1][val-peek] ep {ep}/{config.EPOCHS_SIAMESE} loss~ {np.mean(peek_losses):.4f} (first {VAL_PEEK_STEPS} batches)", flush=True)

        # ---- full validation on EMA ----
        print("[VALIDATION:S1] started")
        siam_eval = SiameseTriplet().to(device).to(memory_format=torch.channels_last)
        ema_siam.load_into(siam_eval)
        clf_eval = HeadBinaryClassifier().to(device)
        ema_aux.load_into(clf_eval)
        siam_eval.eval()
        clf_eval.eval()

        vl, probs, ys, vpos, vneg = [], [], [], [], []
        with torch.no_grad():
            for xa, xp, xn, y in tri_va:
                xa = xa.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last)
                xp = xp.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last)
                xn = xn.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True)
                with torch.autocast(**autocast_kwargs):
                    za = siam_eval.forward_once(xa)
                    zp = siam_eval.forward_once(xp)
                    zn = siam_eval.forward_once(xn)
                    if ep <= supcon_warm:
                        z_cat = torch.cat([za, zp], dim=0)
                        y_cat = torch.cat([y, y], dim=0)
                        loss = supcon_loss(z_cat, y_cat, tau=supcon_tau)
                    else:
                        d_ap = d_cos_pair(za, zp)
                        d_an = d_cos_pair(za, zn)
                        trip = torch.relu(d_ap - d_an + margin).mean()
                        logits = clf_eval(za)
                        ce = ce_aux(logits, y)
                        loss = trip + aux_lambda * ce
                vl.append(loss.item())
                if ep > supcon_warm:
                    probs.append(torch.softmax(logits, dim=1)
                                 [:, 1].float().cpu())
                    ys.append(y.cpu())
                    vpos.append(d_ap.mean().item())
                    vneg.append(d_an.mean().item())

        va_loss = float(np.mean(vl))
        s1_va_loss.append(va_loss)
        if ep <= supcon_warm:
            s1_va_acc.append(np.nan)
            s1_va_auc.append(np.nan)
            s1_va_pos.append(np.nan)
            s1_va_neg.append(np.nan)
        else:
            va_p = torch.cat(probs).numpy()
            va_y = torch.cat(ys).numpy()
            s1_va_acc.append(((va_p > 0.5).astype(int) == va_y).mean())
            try:
                s1_va_auc.append(roc_auc_score(va_y, va_p))
            except Exception:
                s1_va_auc.append(0.5)
            s1_va_pos.append(float(np.mean(vpos)))
            s1_va_neg.append(float(np.mean(vneg)))
        print("[VALIDATION:S1] finished")

        msg_dist = (
            f" | mean_distance_pos/neg(val) {s1_va_pos[-1]:.3f}/{s1_va_neg[-1]:.3f}"
            if not np.isnan(s1_va_pos[-1]) and not np.isnan(s1_va_neg[-1]) else ""
        )

        print(
            f"[S1] epoch {ep}/{config.EPOCHS_SIAMESE} | "
            f"loss train/val {s1_tr_loss[-1]:.4f}/{s1_va_loss[-1]:.4f} | "
            f"accuracy train/val {s1_tr_acc[-1]:.3f}/{s1_va_acc[-1]:.3f} | "
            f"auc_roc train/val {s1_tr_auc[-1]:.3f}/{s1_va_auc[-1]:.3f}{msg_dist}",
            flush=True
        )

        sched.step()
        if s1_va_loss[-1] < best_val - 1e-4:
            best_val, best_ep = s1_va_loss[-1], ep
            torch.save(ema_siam.shadow, os.path.join(
                config.ARTIFACTS, "siamese_stage1_ema.pt"))
            torch.save(ema_aux.shadow,  os.path.join(
                config.ARTIFACTS, "aux_head_ema.pt"))
        elif ep - best_ep >= 5:
            print(f"[S1] early stop at epoch {ep}")
            break

    print("[TRAIN:S1] finished")
    # ensure EMA saved even if no improvement
    torch.save(ema_siam.shadow, os.path.join(
        config.ARTIFACTS, "siamese_stage1_ema.pt"))
    torch.save(ema_aux.shadow,  os.path.join(
        config.ARTIFACTS, "aux_head_ema.pt"))

    if config.SAVE_PLOTS:
        plot_curves(s1_tr_loss, s1_va_loss, "Siamese Network Loss (train vs val)",
                    config.ARTIFACTS, "siamese_loss.png", smooth_k=smooth_k)
        plot_curve(s1_tr_acc, "Classification - training (Siamese)",
                   config.ARTIFACTS, "siamese_classification_train.png", smooth_k=smooth_k, ylabel="accuracy")
        plot_curve(s1_va_acc, "Classification - validation (Siamese)",
                   config.ARTIFACTS, "siamese_classification_val.png", smooth_k=smooth_k, ylabel="accuracy")

    # ---- Stage 2: cached embeddings + classifier ----
    print("[TRAIN:S2] started (classifier on cached embeddings)")
    siam = SiameseTriplet().to(device).to(memory_format=torch.channels_last)
    siam.load_state_dict(torch.load(os.path.join(
        config.ARTIFACTS, "siamese_stage1_ema.pt"), map_location=device))

    Xtr, ytr = cache_embeddings(siam, base_tr, device, autocast_kwargs)
    Xva, yva = cache_embeddings(siam, base_va, device, autocast_kwargs)
    Xte, yte = cache_embeddings(siam, base_te, device, autocast_kwargs)
    print(f"[S2] cached: train={len(Xtr)} val={len(Xva)} test={len(Xte)}")

    # --- NEW: t-SNE scatter on TRAIN embeddings ---
    plot_feature_scatter_2d(Xtr, ytr, os.path.join(config.ARTIFACTS, "tsne_train_embeddings.png"),
                            title="Train embeddings (t-SNE)", max_points=2000)

    # Big batches (embeddings are small)
    bs2 = min(2048, len(Xtr))
    tr = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs2, shuffle=True,
                    pin_memory=True, num_workers=0)
    va = DataLoader(TensorDataset(Xva, yva), batch_size=bs2, shuffle=False,
                    pin_memory=True, num_workers=0)

    clf = HeadBinaryClassifier(in_dim=Xtr.shape[1]).to(device)

    cls_counts = torch.bincount(ytr, minlength=2).float()
    class_weights = (cls_counts.sum() / (cls_counts + 1e-6)).to(device)
    ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    opt = Adam(clf.parameters(), lr=config.LR_CLASSIFIER,
               betas=config.BETAS, weight_decay=1e-4)
    plateau = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    trL, vaL, trA, vaA, trU, vaU = [], [], [], [], [], []
    LOG_EVERY = 10

    best_t_for_acc = 0.5

    for ep in range(1, config.EPOCHS_CLASSIFIER + 1):
        clf.train()
        tl = []
        probs = []
        ys = []
        running2 = {"loss": 0.0, "n": 0}
        for b_idx, (xb, yb) in enumerate(tr, start=1):
            # embeddings are 2D → no channels_last
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.autocast(**autocast_kwargs):
                logits = clf(xb)
                loss = ce(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), config.MAX_NORM)
            opt.step()
            tl.append(loss.item())
            running2["loss"] += float(loss.item())
            running2["n"] += 1
            if running2["n"] % LOG_EVERY == 0:
                avg = running2["loss"] / running2["n"]
                try:
                    total_batches = len(tr)
                except Exception:
                    total_batches = running2["n"]
                print(
                    f"[S2][train] ep {ep}/{config.EPOCHS_CLASSIFIER} b {running2['n']}/{total_batches} | loss {avg:.4f}", flush=True)
            with torch.no_grad():
                probs.append(torch.softmax(logits, dim=1)[:, 1].float().cpu())
                ys.append(yb.cpu())
        trL.append(float(np.mean(tl)))
        p = torch.cat(probs).numpy()
        y = torch.cat(ys).numpy()
        # FIXED: real accuracy
        trA.append(((p >= 0.5).astype(int) == y).mean())
        try:
            trU.append(roc_auc_score(y, p))
        except Exception:
            trU.append(0.5)

        clf.eval()
        vl = []
        probs = []
        ys = []
        with torch.no_grad():
            for xb, yb in va:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.autocast(**autocast_kwargs):
                    logits = clf(xb)
                    vl.append(ce(logits, yb).item())
                probs.append(torch.softmax(logits, dim=1)[:, 1].float().cpu())
                ys.append(yb.cpu())
        vaL.append(float(np.mean(vl)))
        p = torch.cat(probs).numpy()
        y = torch.cat(ys).numpy()
        # FIXED: real accuracy
        vaA.append(((p >= 0.5).astype(int) == y).mean())
        try:
            vaU.append(roc_auc_score(y, p))
        except Exception:
            vaU.append(0.5)

        t_star, vaAccStar = pick_threshold_max_accuracy(y, p, steps=400)
        print(
            f"[S2] best threshold on val for ACC: t*={t_star:.3f} | acc@t*={vaAccStar:.3f}")
        best_t_for_acc = t_star

        prev_lr = opt.param_groups[0]['lr']
        plateau.step(vaU[-1])
        new_lr = opt.param_groups[0]['lr']
        if new_lr < prev_lr:
            print(
                f"[S2] lr reduced: {prev_lr:.2e} -> {new_lr:.2e} (val auc_roc={vaU[-1]:.3f})")

        print(f"[S2] epoch {ep}/{config.EPOCHS_CLASSIFIER} | "
              f"loss train/val {trL[-1]:.4f}/{vaL[-1]:.4f} | "
              f"accuracy train/val {trA[-1]:.3f}/{vaA[-1]:.3f} | "
              f"auc_roc train/val {trU[-1]:.3f}/{vaU[-1]:.3f}")

    if config.SAVE_PLOTS:
        plot_curves(trL, vaL, "Binary Classification Loss (train vs val)",
                    config.ARTIFACTS, "binary_classification_loss.png", smooth_k=smooth_k)

    # ---- test ----
    print("[TEST] started")
    clf.eval()
    with torch.no_grad():
        with torch.autocast(**autocast_kwargs):
            # embeddings are 2D
            logits = clf(Xte.to(device, non_blocking=True))
        probs = torch.softmax(logits, dim=1)[:, 1].to(torch.float32).cpu()
        preds = logits.argmax(1).cpu()

    t = best_t_for_acc
    preds_t = (probs.numpy() >= t).astype(int)
    acc_t = (preds_t == yte.numpy()).mean()
    print(f"[TEST] acc@t* ({t:.3f}) = {acc_t:.3f}")

    # --- NEW: prediction grid on first K original test images (eval tfm) ---
    K = min(64, len(base_te.dataset))
    indices = random.sample(range(len(base_te.dataset)), k=K)
    subset = Subset(base_te.dataset, indices)
    small_loader = DataLoader(subset, batch_size=K,
                              shuffle=False, num_workers=0, pin_memory=True)
    xb_vis, yb_vis = next(iter(small_loader))
    with torch.no_grad():
        with torch.autocast(**autocast_kwargs):
            z_vis = siam.forward_once(xb_vis.to(device, non_blocking=True))
            logits_vis = clf(z_vis)
            probs_vis = torch.softmax(logits_vis, dim=1)[:, 1].float().cpu()
            preds_vis = logits_vis.argmax(1).cpu()
    plot_prediction_grid(xb_vis, yb_vis, probs_vis, preds_vis,
                         os.path.join(config.ARTIFACTS,
                                      "prediction_grid_test.png"),
                         title="Test predictions (first batch)")

    if config.SAVE_PLOTS:
        # NOTE: plot_confusion_matrix now supports colorbar if you updated utils; if not, it still works.
        plot_confusion_matrix(yte.cpu(), preds, ["Benign", "Malignant"],
                              os.path.join(config.ARTIFACTS, "confusion_matrix.png"))
        plot_confusion_matrix(yte.cpu(), torch.tensor(preds_t), ["Benign", "Malignant"],
                              os.path.join(config.ARTIFACTS, "confusion_matrix_test_acc_tstar.png"))
        plot_roc_curve(yte.cpu(), probs, os.path.join(
            config.ARTIFACTS, "roc_curve.png"))

    test_acc = accuracy(clf, Xte, yte, device)
    print(f"[TEST] accuracy {test_acc:.4f}")
    print("[TEST] finished")
    print("[TRAIN:S2] finished")

    if getattr(config, "SAVE_MODELS", True):
        torch.save(siam.state_dict(), os.path.join(
            config.ARTIFACTS, "siamese_final.pt"))
        torch.save(clf.state_dict(),  os.path.join(
            config.ARTIFACTS, "classifier.pt"))


if __name__ == "__main__":
    train()
