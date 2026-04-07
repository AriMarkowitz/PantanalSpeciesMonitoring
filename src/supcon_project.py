"""Stage 1.5: Supervised contrastive projection of Perch embeddings.

Trains a linear projection head using SupCon loss on labeled segments.
The learned projection maps 1536-D Perch embeddings into a space where
same-species embeddings are pulled together and different-species pushed
apart. The downstream clustering (Stage 2) then operates in this
contrastively-shaped space instead of raw Perch space.

Usage:
    python src/supcon_project.py
    python src/supcon_project.py --set supcon.proj_dim=512 --set supcon.epochs=100
"""

import argparse
import os
import sys
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from config import get_config
from utils import setup_logging, build_label_map

_IN_TTY = sys.stdout.isatty()


# ─────────────────────────────────────────────
# SupCon Loss
# ─────────────────────────────────────────────

class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al. 2020).

    For each anchor, positives are other samples with the same species label.
    Negatives are all other samples in the batch. Supports multi-label by
    treating each positive label independently.

    Args:
        temperature: scaling temperature (lower = sharper)
        base_temperature: normalization constant
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) L2-normalized projected embeddings
            labels: (B,) integer species labels (single-label per sample)

        Returns:
            scalar loss
        """
        device = features.device
        B = features.shape[0]

        # Similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # Mask: positive pairs (same label, excluding self)
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        self_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = label_eq & self_mask  # (B, B)

        # For numerical stability
        logits_max, _ = sim.detach().max(dim=1, keepdim=True)
        logits = sim - logits_max

        # Log-sum-exp over all negatives + positives (excluding self)
        exp_logits = torch.exp(logits) * self_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-10)

        # Mean of log-prob over positive pairs
        n_positives = pos_mask.float().sum(dim=1)  # (B,)
        has_pos = n_positives > 0

        if not has_pos.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob = (pos_mask.float() * log_prob).sum(dim=1) / n_positives.clamp(min=1)

        loss = -(self.base_temperature / self.temperature) * mean_log_prob
        return loss[has_pos].mean()


# ─────────────────────────────────────────────
# Projection Head
# ─────────────────────────────────────────────

class ContrastiveProjection(nn.Module):
    """Linear projection: 1536 -> proj_dim with L2 normalization.

    Keeps it linear so the projected space is interpretable and
    the transformation can be applied as a simple matrix multiply
    at inference / clustering time.
    """

    def __init__(self, input_dim: int = 1536, proj_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(input_dim, proj_dim, bias=False)
        # Initialize near-orthogonal
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        return F.normalize(z, dim=-1)

    def project_numpy(self, x: np.ndarray) -> np.ndarray:
        """Project numpy array (no grad, CPU). For use in clustering."""
        with torch.no_grad():
            t = torch.from_numpy(x.astype(np.float32))
            z = self.forward(t)
            return z.numpy()


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class LabeledEmbeddingDataset(Dataset):
    """Loads labeled global embeddings from the embeddings HDF5.

    Only includes segments with a single primary label (XC/iNat focal
    recordings). Soundscapes with multi-label are excluded to keep
    the contrastive objective clean.
    """

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = embeddings.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.from_numpy(self.embeddings[idx]), self.labels[idx]


def load_labeled_embeddings(cfg: dict, logger):
    """Load global embeddings for labeled single-species segments.

    Returns:
        embeddings: (N, 1536) float32
        labels: (N,) int64 species indices
        label_map: {species_id: index}
    """
    import pandas as pd

    segments_csv = cfg["outputs"]["segments_csv"]
    h5_path = cfg["outputs"]["embeddings_h5"]
    label_map = build_label_map(cfg["data"]["taxonomy_csv"])

    segments = pd.read_csv(segments_csv, low_memory=False)
    h5f = h5py.File(h5_path, "r")
    written = h5f["written"][:]
    global_emb = h5f["global_embeddings"]

    # Select single-label segments (strong_primary quality)
    valid_indices = []
    valid_labels = []

    for idx, row in segments.iterrows():
        if idx >= len(written) or not written[idx]:
            continue
        quality = row.get("label_quality", "")
        if quality != "strong_primary":
            continue
        primary = str(row["primary_label"]).strip()
        if primary in label_map:
            valid_indices.append(idx)
            valid_labels.append(label_map[primary])

    logger.info(f"Loading {len(valid_indices)} single-label embeddings")
    embeddings = global_emb[sorted(valid_indices)]
    labels = np.array(valid_labels, dtype=np.int64)

    # Also include soundscape labeled segments (use dominant label)
    sc_indices = []
    sc_labels = []
    for idx, row in segments.iterrows():
        if idx >= len(written) or not written[idx]:
            continue
        quality = row.get("label_quality", "")
        if quality != "strong_multilabel":
            continue
        species_list = str(row["primary_label"]).split(";")
        species_list = [s.strip() for s in species_list if s.strip() in label_map]
        if len(species_list) == 1:
            # Single-species soundscape segment — clean label
            sc_indices.append(idx)
            sc_labels.append(label_map[species_list[0]])

    if sc_indices:
        logger.info(f"Adding {len(sc_indices)} single-species soundscape segments")
        sc_emb = global_emb[sorted(sc_indices)]
        embeddings = np.vstack([embeddings, sc_emb])
        labels = np.concatenate([labels, np.array(sc_labels, dtype=np.int64)])

    h5f.close()

    # Species distribution
    unique, counts = np.unique(labels, return_counts=True)
    logger.info(f"Total: {len(labels)} segments, {len(unique)} species")
    logger.info(f"Samples/species: min={counts.min()}, max={counts.max()}, "
                f"median={int(np.median(counts))}")

    return embeddings, labels, label_map


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_supcon(cfg: dict):
    logger = setup_logging("supcon_project", cfg["outputs"]["logs_dir"])
    logger.info("Stage 1.5: Supervised Contrastive Projection")

    scfg = cfg.get("supcon", {})
    proj_dim = scfg.get("proj_dim", 256)
    temperature = scfg.get("temperature", 0.07)
    lr = scfg.get("lr", 1e-3)
    weight_decay = scfg.get("weight_decay", 1e-4)
    epochs = scfg.get("epochs", 80)
    batch_size = scfg.get("batch_size", 512)
    seed = scfg.get("seed", 42)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load data
    embeddings, labels, label_map = load_labeled_embeddings(cfg, logger)

    # Class-balanced sampling: ensures each batch sees diverse species
    class_counts = np.bincount(labels)
    sample_weights = 1.0 / class_counts[labels]
    sample_weights = sample_weights / sample_weights.sum()
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(labels),
        replacement=True,
    )

    dataset = LabeledEmbeddingDataset(embeddings, labels)
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    # Model
    model = ContrastiveProjection(input_dim=1536, proj_dim=proj_dim).to(device)
    criterion = SupConLoss(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logger.info(f"Projection: 1536 -> {proj_dim}")
    logger.info(f"SupCon temperature: {temperature}")
    logger.info(f"Batch size: {batch_size}, Epochs: {epochs}")

    # Wandb
    use_wandb = scfg.get("wandb_project") is not None
    run = None
    if use_wandb:
        try:
            import wandb
            job_id = os.environ.get("SLURM_JOB_ID", "local")
            run = wandb.init(
                project=scfg["wandb_project"],
                name=f"supcon-{job_id}",
                config={"proj_dim": proj_dim, "temperature": temperature,
                        "lr": lr, "epochs": epochs, "batch_size": batch_size},
            )
        except Exception as e:
            logger.warning(f"wandb init failed: {e}")
            use_wandb = False

    # Training loop
    out_dir = Path(cfg["outputs"]["prototypes_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for emb_batch, label_batch in loader:
            emb_batch = emb_batch.to(device)
            label_batch = label_batch.to(device)

            z = model(emb_batch)
            loss = criterion(z, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}  lr={cur_lr:.2e}")

        if use_wandb and run:
            wandb.log({"epoch": epoch + 1, "supcon_loss": avg_loss, "lr": cur_lr})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "proj_dim": proj_dim,
                "input_dim": 1536,
                "loss": avg_loss,
            }, out_dir / "supcon_projection.pt")

    logger.info(f"Training complete. Best loss: {best_loss:.4f}")
    logger.info(f"Projection saved to {out_dir / 'supcon_projection.pt'}")

    # Also save the weight matrix as numpy for easy use in clustering
    model.load_state_dict(
        torch.load(out_dir / "supcon_projection.pt",
                    map_location="cpu")["model_state_dict"]
    )
    W = model.proj.weight.detach().cpu().numpy()  # (proj_dim, 1536)
    np.save(out_dir / "supcon_W.npy", W)
    logger.info(f"Projection matrix W: {W.shape} saved to {out_dir / 'supcon_W.npy'}")

    if run:
        wandb.finish()


if __name__ == "__main__":
    cfg = get_config()
    train_supcon(cfg)
