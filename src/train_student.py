"""Stage 1d: Distill student embedder from Perch teacher targets.

Reads pre-computed Perch embeddings from HDF5 (teacher targets) and trains
EfficientNet-B1-BirdSet to match them via cosine loss (global + spatial)
and optional logit MSE. No Perch forward passes at training time — all
teacher targets are pre-computed in embeddings.h5 (and optionally
distill_embeddings.h5 for the expanded corpus).

Usage:
    python src/train_student.py
    python src/train_student.py --set stage1d.epochs=30
    FOLD=0 python src/train_student.py   # held-out fold for validation
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import h5py

from config import get_config
from student_model import StudentEmbedder, DistillationLoss
from utils import setup_logging, load_audio_segment


# ── Dataset ──────────────────────────────────────────────────────────────────

class DistillDataset(Dataset):
    """Dataset for student distillation.

    Reads pre-cached mel spectrograms from a numpy memmap and teacher
    embeddings from RAM. Zero audio I/O during training — all data is
    pre-computed. Fully fork-safe (no HDF5 handles, no file I/O in workers).

    Args:
        segments_csv: path to segments.csv (or distill_segments.csv)
        embeddings_h5: path to HDF5 with teacher global/spatial/logit targets
        cfg: full config dict
        fold: fold index for train/val split (-1 = use all)
        split: "train" or "val"
        n_folds: total folds
    """

    def __init__(self, segments_csv: str, embeddings_h5: str, cfg: dict,
                 fold: int = 0, split: str = "train", n_folds: int = 5):
        import pandas as pd
        from cache_mels import get_mel_shape

        self.cfg = cfg
        self.sr = cfg["data"]["sample_rate"]
        self.seg_dur = cfg["data"]["segment_duration"]
        self.top_k = cfg["stage1"]["top_k_logits"]
        self.mel_cfg = cfg.get("student_mel", {})

        # Load teacher embeddings into RAM (fork-safe via COW)
        with h5py.File(embeddings_h5, "r") as h5:
            written = h5["written"][:]
            self.global_embs = h5["global_embeddings"][:]       # (N, 1536) float32
            self.has_spatial = "spatial_embeddings" in h5
            self.has_logits = "logit_values" in h5
            if self.has_spatial:
                self.spatial_embs = h5["spatial_embeddings"][:]  # (N, 5, 1536) float32
            if self.has_logits:
                self.logit_vals = h5["logit_values"][:].astype(np.float32)
            folds_arr = (h5["folds"][:] if "folds" in h5
                         else np.arange(len(pd.read_csv(segments_csv, nrows=0).columns)) % n_folds
                         if False else np.arange(len(written)) % n_folds)

        # Load cached mel memmap (float16, instant random access)
        memmap_path = embeddings_h5 + ".mels.npy"
        n_mels, T = get_mel_shape(cfg)
        N = len(written)
        if not Path(memmap_path).exists():
            raise FileNotFoundError(
                f"Mel cache not found: {memmap_path}\n"
                f"Run 'python src/cache_mels.py --distill' first."
            )
        self.mels = np.memmap(memmap_path, dtype=np.float16, mode="r",
                              shape=(N, n_mels, T))

        # Only use segments with valid embeddings
        valid = np.where(written)[0]

        # Fold split
        if fold >= 0 and n_folds > 1:
            if split == "val":
                self.indices = valid[np.isin(valid, np.where(folds_arr == fold)[0])]
            else:
                self.indices = valid[np.isin(valid, np.where(folds_arr != fold)[0])]
        else:
            self.indices = valid

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        # Cached mel: memmap read (~0.01ms, OS page cache)
        mel = torch.from_numpy(self.mels[i].astype(np.float32)).unsqueeze(0)  # (1, n_mels, T)

        # Teacher targets (pre-loaded numpy arrays)
        global_emb = torch.from_numpy(self.global_embs[i])  # (1536,)

        if self.has_spatial:
            spatial_emb = torch.from_numpy(self.spatial_embs[i])  # (5, 1536)
        else:
            spatial_emb = global_emb.unsqueeze(0).expand(5, -1)

        if self.has_logits:
            logit_vals = torch.from_numpy(self.logit_vals[i])  # (top_k,)
        else:
            logit_vals = torch.zeros(self.top_k)

        return mel, global_emb, spatial_emb, logit_vals

    def close(self):
        pass


def wav_to_mel(wav: np.ndarray, sr: int, mel_cfg: dict) -> torch.Tensor:
    """Convert waveform to log-mel spectrogram.

    Uses the same fast numpy pipeline as cache_mels (~3ms per 5s segment).

    Returns:
        mel: (1, n_mels, T) float32 tensor
    """
    from cache_mels import wav_to_mel_np
    mel_np = wav_to_mel_np(wav, sr, mel_cfg)  # (n_mels, T)
    return torch.from_numpy(mel_np).unsqueeze(0).float()  # (1, n_mels, T)


# ── Training ──────────────────────────────────────────────────────────────────

def collate_fn(batch):
    mels, globals_, spatials, logits = zip(*batch)
    return (
        torch.stack(mels),
        torch.stack(globals_),
        torch.stack(spatials),
        torch.stack(logits),
    )


def train_one_epoch(model, loader, optimizer, criterion, device, scaler,
                    logger=None):
    model.train()
    total_loss = 0.0
    breakdown_acc = {}
    n_batches = 0
    n_total = len(loader)
    log_every = max(1, n_total // 10)  # log ~10 times per epoch

    for mels, t_global, t_spatial, t_logits in loader:
        mels = mels.to(device)
        t_global = t_global.to(device)
        t_spatial = t_spatial.to(device)
        t_logits = t_logits.to(device)

        with torch.amp.autocast("cuda", enabled=scaler is not None,
                                 dtype=torch.bfloat16):
            s_global, s_spatial = model(mels, normalize=False)
            # Student logit head (optional — reuse global projection)
            s_logits = None  # not adding a logit head to keep model simple
            loss, breakdown = criterion(s_global, t_global,
                                        s_spatial, t_spatial,
                                        s_logits, t_logits)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        total_loss += loss.item()
        for k, v in breakdown.items():
            breakdown_acc[k] = breakdown_acc.get(k, 0.0) + v
        n_batches += 1

        if logger and n_batches % log_every == 0:
            avg_loss = total_loss / n_batches
            logger.info(f"  batch {n_batches}/{n_total}  "
                        f"running_loss={avg_loss:.4f}")
            for h in logger.handlers:
                h.flush()

    avg = {k: v / max(n_batches, 1) for k, v in breakdown_acc.items()}
    avg["loss_total"] = total_loss / max(n_batches, 1)
    return avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    cos_sims_global = []
    n_batches = 0

    for mels, t_global, t_spatial, t_logits in loader:
        mels = mels.to(device)
        t_global = t_global.to(device)
        t_spatial = t_spatial.to(device)
        t_logits = t_logits.to(device)

        s_global, s_spatial = model(mels, normalize=True)
        t_global_n = torch.nn.functional.normalize(t_global, dim=-1)

        loss, _ = criterion(s_global, t_global_n,
                            s_spatial,
                            torch.nn.functional.normalize(t_spatial, dim=-1))

        cos_sim = torch.nn.functional.cosine_similarity(
            s_global, t_global_n, dim=-1
        ).mean().item()
        cos_sims_global.append(cos_sim)
        total_loss += loss.item()
        n_batches += 1

    return (total_loss / max(n_batches, 1),
            np.mean(cos_sims_global) if cos_sims_global else 0.0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(cfg: dict):
    logger = setup_logging("train_student", cfg["outputs"]["logs_dir"])
    scfg = cfg.get("stage1d", {})

    fold = int(os.environ.get("FOLD", 0))
    seed = int(os.environ.get("SEED", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    logger.info(f"Stage 1d: Student Distillation — fold {fold}, device {device}")

    # Datasets
    ds_main = DistillDataset(
        cfg["outputs"]["segments_csv"],
        cfg["outputs"]["embeddings_h5"],
        cfg, fold=fold, split="train",
    )
    val_ds = DistillDataset(
        cfg["outputs"]["segments_csv"],
        cfg["outputs"]["embeddings_h5"],
        cfg, fold=fold, split="val",
    )
    logger.info("Embeddings + cached mels loaded (zero I/O during training)")

    # Optionally add distill corpus (no fold splitting — always in train)
    distill_h5 = cfg["outputs"].get("distill_embeddings_h5", "")
    distill_seg = cfg["outputs"].get("distill_segments_csv", "")
    if distill_h5 and Path(distill_h5).exists() and Path(distill_seg).exists():
        ds_distill = DistillDataset(
            distill_seg, distill_h5, cfg, fold=-1, split="train",
        )
        train_ds = ConcatDataset([ds_main, ds_distill])
        logger.info(f"Train: {len(ds_main)} main + {len(ds_distill)} distill = {len(train_ds)}")
    else:
        train_ds = ds_main
        logger.info(f"Train: {len(train_ds)} (no distill corpus found)")

    logger.info(f"Val:   {len(val_ds)}")

    batch_size = scfg.get("batch_size", 64)
    num_workers = scfg.get("num_workers", 4)
    prefetch = scfg.get("prefetch_factor", 2)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, prefetch_factor=prefetch,
                          collate_fn=collate_fn, pin_memory=True,
                          drop_last=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, prefetch_factor=prefetch,
                        collate_fn=collate_fn, pin_memory=True,
                        persistent_workers=True)

    # Model
    logger.info("Loading EfficientNet-B1-BirdSet backbone...")
    model = StudentEmbedder.from_pretrained()
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,}")

    # Loss
    criterion = DistillationLoss(
        lambda_global=scfg.get("lambda_global", 1.0),
        lambda_spatial=scfg.get("lambda_spatial", 1.0),
        lambda_logit=scfg.get("lambda_logit", 0.15),
        use_logit_loss=False,  # no logit head on student — keep it simple
    )

    # Optimizer — lower lr for backbone, higher for heads
    backbone_params = list(model.backbone.parameters())
    head_params = (list(model.global_head.parameters())
                   + list(model.spatial_proj.parameters()))
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": scfg.get("backbone_lr", 1e-5)},
        {"params": head_params,     "lr": scfg.get("head_lr", 1e-4)},
    ], weight_decay=scfg.get("weight_decay", 1e-4))

    max_epochs = scfg.get("epochs", 25)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Wandb
    use_wandb = scfg.get("wandb_project") is not None
    run = None
    if use_wandb:
        try:
            import wandb
            job_id = os.environ.get("SLURM_JOB_ID", "local")
            run = wandb.init(
                project=scfg["wandb_project"],
                name=f"student-{job_id}_fold{fold}",
                config={"fold": fold, "n_params": n_params, **scfg},
            )
        except Exception as e:
            logger.warning(f"wandb init failed: {e}")
            use_wandb = False

    # Checkpointing
    ckpt_dir = Path(cfg["outputs"]["checkpoints_dir"]) / "student"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_cos = 0.0
    best_epoch = -1
    cos_threshold = scfg.get("target_cosine_sim", 0.85)

    for epoch in range(max_epochs):
        train_metrics = train_one_epoch(model, train_dl, optimizer,
                                         criterion, device, scaler,
                                         logger=logger)
        val_loss, val_cos = evaluate(model, val_dl, criterion, device)
        scheduler.step()

        lr_backbone = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]

        logger.info(
            f"Epoch {epoch+1:3d}/{max_epochs}  "
            f"train_loss={train_metrics['loss_total']:.4f}  "
            f"(global={train_metrics.get('loss_global', 0):.4f}  "
            f"spatial={train_metrics.get('loss_spatial', 0):.4f})  "
            f"val_loss={val_loss:.4f}  "
            f"val_cos_sim={val_cos:.4f}  "
            f"lr_backbone={lr_backbone:.2e}  lr_head={lr_head:.2e}"
        )

        if use_wandb and run:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_cosine_sim": val_cos,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                "lr_backbone": lr_backbone,
                "lr_head": lr_head,
            })

        if val_cos > best_cos:
            best_cos = val_cos
            best_epoch = epoch + 1
            job_id = os.environ.get("SLURM_JOB_ID", "local")
            ckpt_path = ckpt_dir / f"{job_id}_fold{fold}_cos={val_cos:.4f}_ep{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_cosine_sim": val_cos,
                "val_loss": val_loss,
                "fold": fold,
                "config": scfg,
            }, ckpt_path)
            logger.info(f"  → Best cos_sim={val_cos:.4f} → {ckpt_path}")

            if val_cos >= cos_threshold:
                logger.info(f"  Target cosine similarity {cos_threshold} reached — consider early stopping")

    logger.info(f"Training complete. Best val_cos={best_cos:.4f} at epoch {best_epoch}")
    if run:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
