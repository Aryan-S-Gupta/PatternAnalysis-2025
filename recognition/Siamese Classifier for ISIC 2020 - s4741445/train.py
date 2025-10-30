import os
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

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


# --- helpers ---

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

    # ---- models ----
    siam = SiameseTriplet().to(device).to(memory_format=torch.channels_last)
    clf_aux = HeadBinaryClassifier().to(device)

    # (Compile disabled on Colab to avoid first-run stalls)
    # try:
    #     siam = torch.compile(siam, mode="reduce-overhead", fullgraph=False)
    #     clf_aux = torch.compile(clf_aux, mode="reduce-overhead", fullgraph=False)
    # except Exception:
    #     pass

    # warmup freeze
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
            s1_tr_acc.append((tr_p > 0.5).astype(int).mean())
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
            s1_va_acc.append((va_p > 0.5).astype(int).mean())
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
        plot_curve(s1_va_acc, "Classification - testing (Siamese)",
                   config.ARTIFACTS, "siamese_classification_test.png", smooth_k=smooth_k, ylabel="accuracy")

    # ---- Stage 2: cached embeddings + classifier ----
    print("[TRAIN:S2] started (classifier on cached embeddings)")
    siam = SiameseTriplet().to(device).to(memory_format=torch.channels_last)
    siam.load_state_dict(torch.load(os.path.join(
        config.ARTIFACTS, "siamese_stage1_ema.pt"), map_location=device))

    Xtr, ytr = cache_embeddings(siam, base_tr, device, autocast_kwargs)
    Xva, yva = cache_embeddings(siam, base_va, device, autocast_kwargs)
    Xte, yte = cache_embeddings(siam, base_te, device, autocast_kwargs)
    print(f"[S2] cached: train={len(Xtr)} val={len(Xva)} test={len(Xte)}")

    # Big batches (embeddings are small) and no workers (TensorDataset is RAM-resident)
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

    for ep in range(1, config.EPOCHS_CLASSIFIER + 1):
        clf.train()
        tl = []
        probs = []
        ys = []
        running2 = {"loss": 0.0, "n": 0}
        for b_idx, (xb, yb) in enumerate(tr, start=1):
            # 2D embeddings → no channels_last
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
        vaA.append(((p >= 0.5).astype(int) == y).mean())
        try:
            vaU.append(roc_auc_score(y, p))
        except Exception:
            vaU.append(0.5)

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
            # 2D embeddings → keep contiguous
            logits = clf(Xte.to(device, non_blocking=True))
        probs = torch.softmax(logits, dim=1)[:, 1].to(torch.float32).cpu()
        preds = logits.argmax(1).cpu()

    if config.SAVE_PLOTS:
        plot_confusion_matrix(yte.cpu(), preds, ["Benign", "Malignant"],
                              os.path.join(config.ARTIFACTS, "confusion_matrix.png"))
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
