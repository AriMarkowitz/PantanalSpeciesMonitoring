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
from augment import spec_augment, mixup_pair


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
                 fold: int = 0, split: str = "train", n_folds: int = 5,
                 augment: bool = False,
                 noise_segment_meta: dict | None = None,
                 live_mels: bool = False):
        """
        Args:
            live_mels: if True, decode audio + compute mels per __getitem__
                (slower but avoids the ~46GB mel memmap on disk). DataLoader
                workers parallelise the decode so GPU stays fed.
            noise_segment_meta: optional {"source_files": np.ndarray,
                "start_secs": np.ndarray, "end_secs": np.ndarray} for
                live noise overlay. Ignored if live_mels=False — in cached
                mode, callers should pass `noise_mel_indices` instead.
        """
        import pandas as pd
        from cache_mels import get_mel_shape

        self.cfg = cfg
        self.sr = cfg["data"]["sample_rate"]
        self.seg_dur = cfg["data"]["segment_duration"]
        self.top_k = cfg["stage1"]["top_k_logits"]
        self.mel_cfg = cfg.get("student_mel", {})
        self.augment = augment
        self.live_mels = live_mels
        # Two parallel paths for noise overlay:
        #   live_mels=True  → noise_segment_meta (file paths + offsets)
        #   live_mels=False → noise_mel_indices (rows in the mels memmap)
        self.noise_segment_meta = noise_segment_meta
        self.noise_mel_indices = (noise_segment_meta or {}).get("indices")

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

        from cache_mels import _resolve_mels_path
        n_mels, T = get_mel_shape(cfg)
        N = len(written)
        memmap_path = _resolve_mels_path(embeddings_h5)
        self.mels: np.memmap | None = None

        if self.live_mels:
            # Live-build path: keep segment audio metadata for on-the-fly mel.
            seg = pd.read_csv(segments_csv, low_memory=False,
                               usecols=["source_file", "start_sec", "end_sec"])
            self.source_files = seg["source_file"].astype(str).values
            self.start_secs = seg["start_sec"].astype(np.float64).values
            self.end_secs = seg["end_sec"].astype(np.float64).values
        elif Path(memmap_path).exists():
            self.mels = np.memmap(memmap_path, dtype=np.float16, mode="r",
                                   shape=(N, n_mels, T))
        else:
            raise FileNotFoundError(
                f"Mel cache not found: {memmap_path}\n"
                f"Either rebuild with `python src/cache_mels.py`, or set "
                f"stage1d.live_mels=true in config to compute mels on the fly."
            )

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

    def _mel_for_index(self, i: int) -> torch.Tensor:
        """Return (1, n_mels, T) float32 mel for segment i.

        Cached path: float16 memmap read (~0.01ms).
        Live path:   audio decode (~5-20ms) + mel STFT (~3ms).
        """
        if self.live_mels:
            from cache_mels import wav_to_mel_np
            wav = load_audio_segment(
                str(self.source_files[i]),
                float(self.start_secs[i]),
                float(self.end_secs[i]),
                sr=self.sr,
            )
            mel_np = wav_to_mel_np(wav, self.sr, self.mel_cfg)  # (n_mels, T)
            return torch.from_numpy(mel_np).unsqueeze(0).float()
        return torch.from_numpy(self.mels[i].astype(np.float32)).unsqueeze(0)

    def _sample_noise_mel(self) -> torch.Tensor | None:
        """Pick a random noise mel for overlay (cached or live path)."""
        meta = self.noise_segment_meta or {}
        # Live path: noise_segment_meta provides full audio metadata
        if self.live_mels and meta.get("source_files") is not None:
            files = meta["source_files"]
            starts = meta["start_secs"]
            ends = meta["end_secs"]
            if len(files) == 0:
                return None
            j = int(torch.randint(0, len(files), (1,)).item())
            try:
                wav = load_audio_segment(
                    str(files[j]), float(starts[j]), float(ends[j]), sr=self.sr,
                )
            except Exception:
                return None
            from cache_mels import wav_to_mel_np
            mel_np = wav_to_mel_np(wav, self.sr, self.mel_cfg)
            return torch.from_numpy(mel_np).unsqueeze(0).float()
        # Cached path: noise_mel_indices points into the memmap
        if (not self.live_mels and self.noise_mel_indices is not None
                and len(self.noise_mel_indices) > 0):
            ni = int(self.noise_mel_indices[
                torch.randint(0, len(self.noise_mel_indices), (1,)).item()
            ])
            return torch.from_numpy(self.mels[ni].astype(np.float32)).unsqueeze(0)
        return None

    def __getitem__(self, idx):
        i = self.indices[idx]

        mel = self._mel_for_index(i)  # (1, n_mels, T)

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

        # ── Student-side augmentation (training only) ───────────────────────
        # Teacher targets stay frozen — we only distort what the student sees.
        if self.augment:
            if torch.rand(1).item() < 0.5:
                noise_mel = self._sample_noise_mel()
                if noise_mel is not None:
                    sig_p = mel.pow(2).mean() + 1e-10
                    noi_p = noise_mel.pow(2).mean() + 1e-10
                    snr_db = float(torch.empty(1).uniform_(0.0, 20.0).item())
                    scale = torch.sqrt(
                        sig_p / (noi_p * (10.0 ** (snr_db / 10.0)))
                    )
                    mel = mel + scale * noise_mel
                    lo, hi = mel.min(), mel.max()
                    mel = (mel - lo) / (hi - lo + 1e-8)

            mel = spec_augment(mel, time_mask_pct=0.2, freq_mask_pct=0.2,
                                num_time_masks=2, num_freq_masks=2)

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
                    mixup_alpha: float = 0.0, use_logit_loss: bool = False,
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

        # Batch-level mixup on the student input; teacher targets are
        # convex-combined with the same lambda.
        if mixup_alpha > 0:
            mels, t_global, t_spatial, t_logits, _lam = mixup_pair(
                mels, t_global, t_spatial, t_logits, alpha=mixup_alpha,
            )

        with torch.amp.autocast("cuda", enabled=scaler is not None,
                                 dtype=torch.bfloat16):
            if use_logit_loss:
                s_global, s_spatial, s_logits = model(
                    mels, normalize=False, return_logits=True,
                )
            else:
                s_global, s_spatial = model(mels, normalize=False)
                s_logits = None
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

    live_mels = bool(scfg.get("live_mels", False))
    logger.info(f"Mel mode: {'LIVE (per-sample audio decode)' if live_mels else 'CACHED memmap'}")

    # ── Noise pool ─────────────────────────────────────────────────────────
    # Unlabeled soundscape segments → mel overlay at a random SNR.
    # Live path keeps audio metadata; cached path keeps mel-row indices.
    import pandas as pd
    _seg = pd.read_csv(
        cfg["outputs"]["segments_csv"], low_memory=False,
        usecols=["source_file", "start_sec", "end_sec",
                  "source_type", "label_quality"],
    )
    noise_mask = (
        (_seg["source_type"].astype(str).values == "soundscape")
        & (_seg["label_quality"].astype(str).values == "unlabeled")
    )
    noise_idx_full = np.where(noise_mask)[0]
    if live_mels:
        # Cap the noise candidate pool: 127k is overkill, and full file
        # paths held in worker memory add up. 5k is plenty for variety.
        cap = min(5000, len(noise_idx_full))
        if len(noise_idx_full) > cap:
            rng_pick = np.random.default_rng(42)
            noise_idx_full = rng_pick.choice(noise_idx_full, size=cap, replace=False)
        noise_segment_meta = {
            "source_files": _seg["source_file"].astype(str).values[noise_idx_full],
            "start_secs": _seg["start_sec"].astype(np.float64).values[noise_idx_full],
            "end_secs": _seg["end_sec"].astype(np.float64).values[noise_idx_full],
        }
        logger.info(f"Noise pool (live, sampled from unlabeled soundscape): "
                    f"{len(noise_idx_full)}")
    else:
        noise_segment_meta = {"indices": noise_idx_full}
        logger.info(f"Noise pool (cached mels, unlabeled soundscape): "
                    f"{len(noise_idx_full)}")

    # Datasets
    augment_train = bool(scfg.get("augment", True))
    ds_main = DistillDataset(
        cfg["outputs"]["segments_csv"],
        cfg["outputs"]["embeddings_h5"],
        cfg, fold=fold, split="train",
        augment=augment_train, noise_segment_meta=noise_segment_meta,
        live_mels=live_mels,
    )
    val_ds = DistillDataset(
        cfg["outputs"]["segments_csv"],
        cfg["outputs"]["embeddings_h5"],
        cfg, fold=fold, split="val",
        augment=False, live_mels=live_mels,
    )
    logger.info(
        f"Embeddings loaded (zero embedding I/O during training); "
        f"train augment={augment_train}"
    )

    # Optionally add distill corpus (no fold splitting — always in train)
    distill_h5 = cfg["outputs"].get("distill_embeddings_h5", "")
    distill_seg = cfg["outputs"].get("distill_segments_csv", "")
    if distill_h5 and Path(distill_h5).exists() and Path(distill_seg).exists():
        # Distill corpus has no unlabeled soundscapes; pass an empty meta
        # so only SpecAugment + mixup apply (no noise overlay).
        empty_meta = {} if live_mels else {"indices": np.array([], dtype=np.int64)}
        ds_distill = DistillDataset(
            distill_seg, distill_h5, cfg, fold=-1, split="train",
            augment=augment_train, noise_segment_meta=empty_meta,
            live_mels=live_mels,
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
    use_logit_loss = bool(scfg.get("use_logit_loss", True))
    if use_logit_loss:
        model.attach_logit_head(top_k=cfg["stage1"]["top_k_logits"])
        logger.info(f"Attached logit head (top_k={cfg['stage1']['top_k_logits']})")
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,}")

    # Loss
    criterion = DistillationLoss(
        lambda_global=scfg.get("lambda_global", 1.0),
        lambda_spatial=scfg.get("lambda_spatial", 1.0),
        lambda_logit=scfg.get("lambda_logit", 0.15),
        use_logit_loss=use_logit_loss,
    )

    # Optimizer — lower lr for backbone, higher for heads
    backbone_params = list(model.backbone.parameters())
    head_params = (list(model.global_head.parameters())
                   + list(model.spatial_proj.parameters()))
    if use_logit_loss and model.logit_head is not None:
        head_params += list(model.logit_head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": scfg.get("backbone_lr", 1e-5)},
        {"params": head_params,     "lr": scfg.get("head_lr", 1e-4)},
    ], weight_decay=scfg.get("weight_decay", 1e-4))

    max_epochs = scfg.get("epochs", 25)
    warmup_epochs = scfg.get("warmup_epochs", 5)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs - warmup_epochs
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
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
    patience = scfg.get("early_stop_patience", 10)
    epochs_without_improvement = 0

    mixup_alpha = float(scfg.get("mixup_alpha", 0.2))
    logger.info(f"Augment: specaug={augment_train}  mixup_alpha={mixup_alpha}  "
                f"logit_loss={use_logit_loss}")

    for epoch in range(max_epochs):
        train_metrics = train_one_epoch(model, train_dl, optimizer,
                                         criterion, device, scaler,
                                         mixup_alpha=mixup_alpha,
                                         use_logit_loss=use_logit_loss,
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
            epochs_without_improvement = 0
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
                logger.info(f"  Target cosine similarity {cos_threshold} reached")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"  Early stopping: no improvement for {patience} epochs "
                            f"(best={best_cos:.4f} at epoch {best_epoch})")
                break

    logger.info(f"Training complete. Best val_cos={best_cos:.4f} at epoch {best_epoch}")
    if run:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
