"""Train the prototypical probing head on frozen Student spatial embeddings.

No features.h5, no HDBSCAN, no GMMs. Reads spatial embeddings (N, L, D)
directly from student_embeddings.h5, builds labels/masks from segments.csv,
and trains `PrototypicalHead` end-to-end with masked BCE + orthogonality.

Usage:
    python src/train_proto.py
    python src/train_proto.py --set probe.prototypes_per_class=20
    FOLD=2 python src/train_proto.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import h5py
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from config import get_config
from proto_probe import PrototypicalHead, MaskedBCEWithOrthoLoss
from build_features import build_labels_and_masks, assign_folds
from utils import setup_logging, build_label_map


class SpatialEmbeddingDataset(Dataset):
    """In-memory dataset of (spatial_emb, label, mask) triples.

    Loads the entire spatial_embeddings matrix into RAM (358k × 5 × 1536
    float32 ≈ 11 GB). Cheap on any node with > 16 GB and avoids HDF5
    per-sample overhead. Set `preload=False` to page from disk instead.
    """

    def __init__(self, emb_h5: str, labels: np.ndarray, masks: np.ndarray,
                 indices: np.ndarray, preload: bool = True):
        self.emb_h5 = emb_h5
        self.labels = labels
        self.masks = masks
        self.indices = indices
        self.preload = preload
        if preload:
            with h5py.File(emb_h5, "r") as h5:
                self.spatial = h5["spatial_embeddings"][:]  # (N, L, D)
        else:
            self._h5 = None

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.emb_h5, "r")
        return self._h5

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = int(self.indices[idx])
        if self.preload:
            spatial = self.spatial[i]
        else:
            spatial = self._open()["spatial_embeddings"][i]
        return (
            torch.from_numpy(spatial.astype(np.float32)),
            torch.from_numpy(self.labels[i]),
            torch.from_numpy(self.masks[i]),
        )


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    aucs = []
    for c in range(y_true.shape[1]):
        pos = y_true[:, c].sum()
        if 0 < pos < len(y_true):
            try:
                aucs.append(roc_auc_score(y_true[:, c], y_pred[:, c]))
            except ValueError:
                continue
    return float(np.mean(aucs)) if aucs else 0.0


def train_one_epoch(head, loader, opt, criterion, device):
    head.train()
    total = 0.0
    n = 0
    for spatial, labels, masks in loader:
        spatial = spatial.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = head(spatial)
        loss = criterion(logits, labels, masks, head)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(head, loader, criterion, device):
    head.eval()
    total = 0.0
    n = 0
    all_p = []
    all_y = []
    all_m = []
    for spatial, labels, masks in loader:
        spatial = spatial.to(device, non_blocking=True)
        labels_d = labels.to(device, non_blocking=True)
        masks_d = masks.to(device, non_blocking=True)

        logits = head(spatial)
        loss = criterion(logits, labels_d, masks_d, head)

        total += loss.item()
        n += 1
        all_p.append(torch.sigmoid(logits).cpu().numpy())
        all_y.append(labels.numpy())
        all_m.append(masks.numpy())

    preds = np.concatenate(all_p, axis=0)
    ys = np.concatenate(all_y, axis=0)
    ms = np.concatenate(all_m, axis=0)

    # Macro AUC computed only on supervised rows per class — unsupervised
    # (mask=0) samples shouldn't drive either positive or negative count.
    aucs = []
    for c in range(ys.shape[1]):
        sel = ms[:, c] > 0
        if sel.sum() < 2:
            continue
        pos = ys[sel, c].sum()
        if 0 < pos < sel.sum():
            try:
                aucs.append(roc_auc_score(ys[sel, c], preds[sel, c]))
            except ValueError:
                continue
    return total / max(n, 1), float(np.mean(aucs)) if aucs else 0.0


def main(cfg: dict):
    logger = setup_logging("train_proto", cfg["outputs"]["logs_dir"])
    pcfg = cfg.get("probe", {})

    fold = int(os.environ.get("FOLD", pcfg.get("fold", 0)))
    J = int(pcfg.get("prototypes_per_class", 10))
    lr = float(pcfg.get("lr", 1e-3))
    weight_decay = float(pcfg.get("weight_decay", 1e-4))
    ortho_weight = float(pcfg.get("ortho_weight", 0.1))
    batch_size = int(pcfg.get("batch_size", 256))
    epochs = int(pcfg.get("epochs", 30))
    n_folds = int(cfg["stage4"]["n_folds"])
    num_workers = int(pcfg.get("num_workers", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Training probe on fold {fold}/{n_folds} (device={device}, "
                f"J={J}, epochs={epochs}, bs={batch_size}, lr={lr})")

    # ── Data ────────────────────────────────────────────────────────────────
    segments = pd.read_csv(cfg["outputs"]["segments_csv"], low_memory=False)
    label_map = build_label_map(cfg["data"]["taxonomy_csv"])
    num_species = len(label_map)

    emb_h5 = str(Path(cfg["outputs"]["embeddings_h5"]).parent / "student_embeddings.h5")
    if not Path(emb_h5).exists():
        raise FileNotFoundError(
            f"Student embeddings not found at {emb_h5}. "
            f"Run src/extract_student_embeddings.py first."
        )
    logger.info(f"Embeddings: {emb_h5}")

    with h5py.File(emb_h5, "r") as h5:
        written = h5["written"][:]
        L, D = h5["spatial_embeddings"].shape[1], h5["spatial_embeddings"].shape[2]
    logger.info(f"Spatial embedding shape per segment: ({L}, {D})")

    logger.info("Building labels, masks, folds …")
    labels, masks = build_labels_and_masks(segments, label_map)
    folds = assign_folds(segments, n_folds)

    # Supervised samples with embeddings on this fold
    train_idx = np.where((folds != fold) & (folds >= 0) & written
                          & (masks.sum(axis=1) > 0))[0]
    val_idx = np.where((folds == fold) & written & (masks.sum(axis=1) > 0))[0]
    logger.info(f"train={len(train_idx)}, val={len(val_idx)}")

    train_ds = SpatialEmbeddingDataset(emb_h5, labels, masks, train_idx,
                                        preload=pcfg.get("preload", True))
    val_ds = SpatialEmbeddingDataset(emb_h5, labels, masks, val_idx,
                                      preload=pcfg.get("preload", True))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    # ── Model ──────────────────────────────────────────────────────────────
    head = PrototypicalHead(embed_dim=D, num_classes=num_species,
                             prototypes_per_class=J).to(device)
    logger.info(f"PrototypicalHead: C={num_species}, J={J}, D={D}, "
                f"params={sum(p.numel() for p in head.parameters()):,}")

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = MaskedBCEWithOrthoLoss(ortho_weight=ortho_weight)

    ckpt_dir = Path(cfg["outputs"]["checkpoints_dir"]) / "proto_probe"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0
    start_epoch = 1

    # ── Resume from last checkpoint if requested ────────────────────────────
    # RESUME=1 → resume from fold{FOLD}_last.pt (full state: model+opt+sched).
    # RESUME=auto → resume if last.pt exists, else start fresh.
    # RESUME=path/to/ckpt.pt → resume from a specific file (model state only
    #                          if it's a *_best.pt, otherwise full state).
    resume = os.environ.get("RESUME", "")
    resume_path = None
    if resume == "1":
        resume_path = ckpt_dir / f"fold{fold}_last.pt"
        if not resume_path.exists():
            raise FileNotFoundError(
                f"RESUME=1 but no checkpoint at {resume_path}. "
                f"Run without RESUME first, or pass RESUME=<path> to an existing ckpt."
            )
    elif resume == "auto":
        cand = ckpt_dir / f"fold{fold}_last.pt"
        if cand.exists():
            resume_path = cand
    elif resume:
        resume_path = Path(resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"RESUME path does not exist: {resume_path}")

    if resume_path is not None:
        logger.info(f"Resuming from {resume_path}")
        state = torch.load(resume_path, map_location=device, weights_only=False)
        head.load_state_dict(state["model_state_dict"])
        if "optimizer_state_dict" in state:
            opt.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            sched.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_auc = float(state.get("best_auc", state.get("val_auc", -1.0)))
        logger.info(
            f"  resumed at epoch={start_epoch}, best_auc={best_auc:.4f}  "
            f"(target: {epochs} total epochs)"
        )
        if start_epoch > epochs:
            logger.warning(
                f"start_epoch ({start_epoch}) > epochs ({epochs}); "
                f"raise probe.epochs in config to train further."
            )

    # ── Train ──────────────────────────────────────────────────────────────
    def _save(path: Path, val_auc: float, ep: int):
        torch.save({
            "model_state_dict": head.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "config": {"J": J, "D": D, "num_classes": num_species},
            "val_auc": val_auc,
            "best_auc": best_auc,
            "fold": fold,
            "epoch": ep,
        }, path)

    for ep in range(start_epoch, epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(head, train_dl, opt, criterion, device)
        val_loss, val_auc = evaluate(head, val_dl, criterion, device)
        sched.step()
        logger.info(
            f"epoch {ep:3d}/{epochs}  "
            f"tr_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_auc={val_auc:.4f}  "
            f"({time.time() - t0:.1f}s)"
        )
        if val_auc > best_auc:
            best_auc = val_auc
            _save(ckpt_dir / f"fold{fold}_best.pt", val_auc, ep)
            logger.info(f"  → new best val_auc={val_auc:.4f}, saved")
        # Always checkpoint the latest state for resume
        _save(ckpt_dir / f"fold{fold}_last.pt", val_auc, ep)

    logger.info(f"Fold {fold} done. best_val_auc={best_auc:.4f}")


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
