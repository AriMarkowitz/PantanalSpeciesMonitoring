"""Stage 4b: Pseudo-labeling pipeline.

Two rounds:
  Round 1 — Bootstrap from pre-computed Perch logit pseudo-labels.
             Reads the external pseudo_labels CSVs (filename, start, end, primary_label)
             and converts them to a dense (N, C) array aligned to segments.csv.

  Round 2+ — Self-training: run the trained MLP classifier on unlabeled segments
              in features.h5, threshold per-species predictions, and return a
              (N, C) pseudo-label array for the next training round.

Usage:
    # Round 1: bootstrap from Perch logit pseudo-labels
    python src/pseudo_label.py --round 1

    # Round 2+: self-training from a trained checkpoint
    python src/pseudo_label.py --round 2 --checkpoint outputs/checkpoints/.../best.pt

    # Override threshold
    python src/pseudo_label.py --round 2 --checkpoint ... --set stage4b.self_train_threshold=0.8
"""

from __future__ import annotations

import argparse
import logging
import numpy as np
import pandas as pd
import torch
import h5py
from pathlib import Path

from config import get_config
from model import MotifClassifier
from utils import setup_logging, build_label_map


# ── Round 1: load external Perch-logit pseudo-labels ─────────────────────────

def load_perch_pseudo_labels(csv_paths: list[str],
                             segments: pd.DataFrame,
                             label_map: dict,
                             logger: logging.Logger) -> np.ndarray:
    """Convert Perch-logit pseudo-label CSVs → dense (N, C) array.

    The CSVs have schema: filename, start, end, primary_label
    We match rows to segments.csv by (source_file basename, start_sec).

    Returns:
        pseudo_labels: (N, C) float32, values in {0, 0.5*weight} range.
            0 means no pseudo-label for that cell.
    """
    num_species = len(label_map)
    N = len(segments)
    pseudo_labels = np.zeros((N, num_species), dtype=np.float32)

    # Build lookup: (basename, start_sec_rounded) → segment index
    lookup: dict[tuple[str, float], int] = {}
    for i, row in segments.iterrows():
        basename = Path(str(row["source_file"])).name
        start = round(float(row["start_sec"]), 1)
        lookup[(basename, start)] = i

    n_matched = 0
    n_unknown_species = 0

    for csv_path in csv_paths:
        if not Path(csv_path).exists():
            logger.warning(f"Pseudo-label CSV not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        logger.info(f"  Loading {csv_path}: {len(df)} rows")

        for _, row in df.iterrows():
            basename = Path(str(row["filename"])).name
            # start field is "HH:MM:SS" or seconds float
            start_raw = str(row["start"])
            if ":" in start_raw:
                parts = start_raw.split(":")
                start_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            else:
                start_sec = float(start_raw)
            start_sec = round(start_sec, 1)

            seg_idx = lookup.get((basename, start_sec))
            if seg_idx is None:
                continue

            sp = str(row["primary_label"]).strip()
            if sp not in label_map:
                n_unknown_species += 1
                continue

            pseudo_labels[seg_idx, label_map[sp]] = 1.0
            n_matched += 1

    logger.info(f"  Matched {n_matched} pseudo-labels, "
                f"{n_unknown_species} skipped (species not in taxonomy)")
    return pseudo_labels


# ── Round 2+: self-training inference ────────────────────────────────────────

@torch.no_grad()
def run_self_training(checkpoint_path: str,
                      features_h5: str,
                      threshold: float,
                      logger: logging.Logger) -> np.ndarray:
    """Run trained MLP on all segments, threshold predictions.

    Returns:
        pseudo_labels: (N, C) float32 — 1.0 where prediction > threshold, else 0.0
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Self-training inference on {device}, threshold={threshold}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    feat_dim = ckpt["feat_dim"]
    num_classes = ckpt["num_classes"]
    cfg_model = ckpt.get("config", {})

    model = MotifClassifier(
        input_dim=feat_dim,
        num_classes=num_classes,
        hidden_dims=cfg_model.get("hidden_dims", [512, 256]),
        dropout=0.0,  # no dropout at inference
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"  Loaded checkpoint: val_auc={ckpt.get('val_auc', 'N/A'):.4f}")

    h5f = h5py.File(features_h5, "r")
    features = h5f["features"]
    masks = h5f["masks"][:]
    N = features.shape[0]

    pseudo_labels = np.zeros((N, num_classes), dtype=np.float32)
    batch_size = 4096
    n_pseudo = 0

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = torch.from_numpy(
            features[start:end].astype(np.float32)
        ).to(device)
        probs = torch.sigmoid(model(batch)).cpu().numpy()

        # Only pseudo-label segments that are currently unlabeled
        for i_local, i_global in enumerate(range(start, end)):
            if masks[i_global].sum() > 0:
                continue  # already has real labels — skip
            preds = probs[i_local]
            confident = preds > threshold
            if confident.any():
                pseudo_labels[i_global, confident] = preds[confident]
                n_pseudo += confident.sum()

    h5f.close()
    logger.info(f"  Generated {n_pseudo} pseudo-label entries across "
                f"{(pseudo_labels.sum(axis=1) > 0).sum()} segments")
    return pseudo_labels


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_pseudo_labels(pseudo_labels: np.ndarray, out_path: str,
                       logger: logging.Logger):
    """Save pseudo-label array as .npz."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, pseudo_labels=pseudo_labels)
    logger.info(f"Saved pseudo-labels → {out_path}  "
                f"({(pseudo_labels > 0).sum()} non-zero entries)")


def load_pseudo_labels(path: str) -> np.ndarray:
    """Load pseudo-label array from .npz."""
    return np.load(path)["pseudo_labels"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main(cfg: dict):
    logger = setup_logging("pseudo_label", cfg["outputs"]["logs_dir"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default="")
    args, _ = parser.parse_known_args()

    pl_cfg = cfg["stage4b"]
    out_dir = Path(cfg["outputs"]["pseudo_labels_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"pseudo_labels_r{args.round}.npz")

    logger.info(f"Stage 4b: Pseudo-Labeling — round {args.round}")

    if args.round == 1:
        # Bootstrap from Perch logit pseudo-label CSVs
        segments = pd.read_csv(cfg["outputs"]["segments_csv"], low_memory=False)
        label_map = build_label_map(cfg["data"]["taxonomy_csv"])

        # Collect all available pseudo-label CSV paths from data dir
        data_dir = Path(cfg["data"]["train_csv"]).parent
        csv_candidates = [
            data_dir / "pseudo_labels_sc.csv",
            data_dir / "pseudo_labels_audio.csv",
            data_dir / "pseudo_labels_t06_sc.csv",
            data_dir / "pseudo_labels_t06_audio.csv",
        ]
        csv_paths = [str(p) for p in csv_candidates if p.exists()]
        logger.info(f"Found {len(csv_paths)} pseudo-label CSV(s): {csv_paths}")

        pseudo_labels = load_perch_pseudo_labels(
            csv_paths, segments, label_map, logger
        )

    else:
        # Self-training from trained checkpoint
        if not args.checkpoint:
            raise ValueError("--checkpoint required for round >= 2")

        threshold = pl_cfg["self_train_threshold"]
        pseudo_labels = run_self_training(
            args.checkpoint,
            cfg["outputs"]["features_h5"],
            threshold,
            logger,
        )

    save_pseudo_labels(pseudo_labels, out_path, logger)
    logger.info(f"Stage 4b round {args.round} complete → {out_path}")


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)