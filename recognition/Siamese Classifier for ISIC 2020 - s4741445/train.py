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
from models import SiameseTriplet, HeadBinaryClassifier
from utils import (
    set_seed, plot_curves, accuracy, plot_tsne, plot_confusion_matrix,
    plot_roc_curve, save_distance_hist, save_three_panel
)

torch.backends.cudnn.benchmark = True

# --- helpers ---


def d_cos_pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - (a * b).sum(dim=1).clamp(-1, 1)  # a,b L2-normalized


def supcon_loss(z: torch.Tensor, y: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """
    Supervised NT-Xent: z [N,d] L2-normalized, y [N] labels.
    """
    z = nn.functional.normalize(z, dim=1)
    sim = z @ z.t() / tau                      # [N,N]
    sim = sim - torch.eye(sim.size(0), device=sim.device) * \
        1e9  # mask diagonal
    y = y.view(-1, 1)
    pos_mask = (y == y.t()).float()
    pos_mask.fill_diagonal_(0.0)
    denom = torch.logsumexp(sim, dim=1, keepdim=True)            # [N,1]
    log_prob = sim - denom                                       # [N,N]
    pos_count = pos_mask.sum(1).clamp(min=1.0)
    loss = -(pos_mask * log_prob).sum(1) / pos_count
    return loss.mean()


class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = float(decay)
        # take an initial snapshot (types preserved)
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            sv = self.shadow.get(k, None)
            if sv is None:
                # new key appeared; just clone it
                self.shadow[k] = v.detach().clone()
                continue

            if v.is_floating_point():
                # EMA only for floating tensors
                sv.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                # ints/bools/etc: copy directly (no EMA math)
                sv.copy_(v)

    def load_into(self, model):
        # load the full shadow dict (floats are EMA’d; others are direct copies)
        model.load_state_dict(self.shadow, strict=True)


@torch.no_grad()
def cache_embeddings(siam: SiameseTriplet, loader: DataLoader, device, autocast_kwargs):
    X, y = [], []
    siam.eval()
    for xb, yb in loader:
        xb = xb.to(device)
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

    # ---- models ----
    siam = SiameseTriplet().to(device)
    clf_aux = HeadBinaryClassifier().to(device)

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

    # EMA shadows (evaluate val on EMA for smooth curves)
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
    best_path = os.path.join(config.ARTIFACTS, "siamese_stage1.pt")
    print("Stage 1: multi-task (Triplet + Aux CE) with SupCon warmup + EMA")
