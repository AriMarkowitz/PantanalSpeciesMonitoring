"""Stage 4: Train MLP classifier on pre-computed feature vectors.

Plain PyTorch training loop — no Lightning needed for a 2-layer MLP
on frozen features. Trains in minutes per fold.

Usage:
    python src/train_classifier.py
    python src/train_classifier.py --set stage4.lr=5e-4
    FOLD=2 python src/train_classifier.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score

from config import get_config
from model import MotifClassifier, MaskedFocalLoss, MaskedAsymmetricLoss
from dataset import get_dataloaders
from utils import setup_logging


def compute_macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged ROC-AUC, skipping classes with no positive samples."""
    aucs = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0 and y_true[:, i].sum() < len(y_true):
            try:
                aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
            except ValueError:
                continue
    return np.mean(aucs) if aucs else 0.0


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for feats, labels, masks in loader:
        feats = feats.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        logits = model(feats)
        loss = criterion(logits, labels, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []
    all_masks = []

    for feats, labels, masks in loader:
        feats = feats.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        logits = model(feats)
        loss = criterion(logits, labels, masks)

        total_loss += loss.item()
        n_batches += 1

        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_masks.append(masks.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    masks = np.concatenate(all_masks)

    # Per-species masked AUC: only evaluate each species on rows where
    # that species is supervised (mask > 0), avoiding noise from
    # unsupervised rows whose labels default to zero.
    aucs = []
    for i in range(preds.shape[1]):
        sp_mask = masks[:, i] > 0
        if sp_mask.sum() < 2:
            continue
        y_t = labels[sp_mask, i]
        y_p = preds[sp_mask, i]
        if y_t.sum() > 0 and y_t.sum() < len(y_t):
            try:
                aucs.append(roc_auc_score(y_t, y_p))
            except ValueError:
                continue
    auc = np.mean(aucs) if aucs else 0.0

    return total_loss / max(n_batches, 1), auc


def main(cfg: dict):
    logger = setup_logging("train_classifier", cfg["outputs"]["logs_dir"])

    fold = int(os.environ.get("FOLD", cfg.get("fold", 0)))
    seed = int(os.environ.get("SEED", 42))
    tcfg = cfg["stage4"]

    logger.info(f"Stage 4: Classifier Training — fold {fold}")

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Data
    h5_path = cfg["outputs"]["features_h5"]
    train_dl, val_dl = get_dataloaders(
        h5_path, fold=fold, batch_size=tcfg["batch_size"],
        n_folds=tcfg["n_folds"],
    )
    feat_dim = train_dl.dataset.feat_dim
    num_classes = train_dl.dataset.num_classes
    logger.info(f"Features: {feat_dim}D, Classes: {num_classes}")
    logger.info(f"Train: {len(train_dl.dataset)}, Val: {len(val_dl.dataset)}")

    # Model
    model = MotifClassifier(
        input_dim=feat_dim,
        num_classes=num_classes,
        hidden_dims=tcfg["hidden_dims"],
        dropout=tcfg["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,}")

    # Loss, optimizer, scheduler
    loss_name = tcfg.get("loss", "focal")
    if loss_name == "asl":
        criterion = MaskedAsymmetricLoss(
            gamma_pos=tcfg.get("asl_gamma_pos", 0.0),
            gamma_neg=tcfg.get("asl_gamma_neg", 4.0),
            clip=tcfg.get("asl_clip", 0.05),
        )
        logger.info(f"Loss: ASL (gamma_pos={tcfg.get('asl_gamma_pos', 0.0)}, "
                     f"gamma_neg={tcfg.get('asl_gamma_neg', 4.0)}, clip={tcfg.get('asl_clip', 0.05)})")
    else:
        criterion = MaskedFocalLoss(
            alpha=tcfg["focal_alpha"],
            gamma=tcfg["focal_gamma"],
        )
        logger.info(f"Loss: Focal (alpha={tcfg['focal_alpha']}, gamma={tcfg['focal_gamma']})")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
    )
    scheduler_name = tcfg.get("scheduler", "cosine")
    if scheduler_name == "cosine_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=tcfg.get("restart_period", 10), T_mult=1,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tcfg["max_epochs"],
        )

    # Wandb
    use_wandb = tcfg.get("wandb_project") is not None
    run = None
    if use_wandb:
        try:
            import wandb
            job_id = os.environ.get("SLURM_JOB_ID", "local")
            run_name = f"motif-{job_id}_fold{fold}"
            run = wandb.init(
                project=tcfg["wandb_project"],
                name=run_name,
                config={
                    "fold": fold, "seed": seed,
                    "feat_dim": feat_dim, "num_classes": num_classes,
                    **tcfg,
                },
            )
        except Exception as e:
            logger.warning(f"wandb init failed: {e}")
            use_wandb = False

    # Checkpointing
    ckpt_dir = Path(cfg["outputs"]["checkpoints_dir"])
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_dir = ckpt_dir / f"{job_id}_fold{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_auc = 0.0
    best_epoch = -1
    patience = tcfg.get("early_stop_patience", 5)
    epochs_since_improvement = 0

    # Training loop
    for epoch in range(tcfg["max_epochs"]):
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, device)
        val_loss, val_auc = evaluate(model, val_dl, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        logger.info(f"Epoch {epoch+1:3d}/{tcfg['max_epochs']}  "
                     f"train_loss={train_loss:.4f}  "
                     f"val_loss={val_loss:.4f}  "
                     f"val_auc={val_auc:.4f}  "
                     f"lr={lr:.2e}")

        if use_wandb and run:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_auc": val_auc,
                "lr": lr,
            })

        # Save best
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            epochs_since_improvement = 0
            ckpt_path = run_dir / f"best_val_auc={val_auc:.4f}_epoch={epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_loss": val_loss,
                "fold": fold,
                "feat_dim": feat_dim,
                "num_classes": num_classes,
                "config": tcfg,
            }, ckpt_path)
            logger.info(f"  → New best: val_auc={val_auc:.4f} saved to {ckpt_path}")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                logger.info(
                    f"Early stopping at epoch {epoch+1}: no val_auc improvement "
                    f"for {patience} epochs (best={best_auc:.4f} @ epoch {best_epoch})"
                )
                break

    logger.info(f"Training complete. Best val_auc={best_auc:.4f} at epoch {best_epoch}")

    # Cleanup
    train_dl.dataset.close()
    val_dl.dataset.close()
    if run:
        wandb.finish()


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
