"""NMFk on a sample of student embeddings → global motif dictionary.

Motivation:
  Per-class NMF on mels didn't discriminate species (mean AUC ~0.55). The
  hypothesis is that mel spectrograms share too much low-level structure
  across species. Embeddings from the Perch/Student encoder already live
  in a species-organized space, so motifs learned there should reflect
  sub-call structure (different call types within a species, vocalization
  vs. contact call, etc.) rather than generic time-frequency primitives.

Pipeline:
  1. Sample N student embeddings stratified by species.
  2. ReLU them (NMF needs non-negativity; embeddings can be negative).
  3. Run pyNMFk canonical rank selection over k ∈ [k_min, k_max].
  4. Save dictionary W (d, k) + activations H (k, N_all) for all segments.

Usage:
    python src/nmf_global.py build          # fit dictionary on sample
    python src/nmf_global.py project        # project all segments → H_all
    python src/nmf_global.py all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch

from config import get_config
from utils import setup_logging, build_label_map

from pynmfk import run_nmfk, format_results_table  # installed via `pip install -e ~/pyNMFk`


def _sample_stratified(segments: pd.DataFrame, written: np.ndarray,
                        label_map: dict, per_species: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Sample up to `per_species` strong_primary segments per species (in label_map).

    Returns sorted segment indices (into the embeddings HDF5).
    """
    mask = (segments["label_quality"].values == "strong_primary") & written
    primary = segments["primary_label"].astype(str).values
    sampled = []
    for sp in label_map.keys():
        idxs = np.where(mask & (primary == sp))[0]
        if len(idxs) == 0:
            continue
        if len(idxs) > per_species:
            idxs = rng.choice(idxs, size=per_species, replace=False)
        sampled.append(idxs)
    if not sampled:
        raise RuntimeError("No strong_primary segments found — check segments.csv/written.")
    return np.sort(np.concatenate(sampled))


# ─────────────────────────────────────────────────────────────────────────────
# CLI commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_build(cfg: dict):
    """Fit a global motif dictionary on a stratified embedding sample."""
    logger = setup_logging("nmf_global", cfg["outputs"]["logs_dir"])
    gcfg = cfg["stage_nmf_global"]
    out_dir = Path(cfg["outputs"]["nmf_global_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = torch.cuda.is_available()
    algo = gcfg.get("algo", "hals")
    logger.info(f"NMF solver: pyNMFk ({algo}) on {'GPU ' + torch.cuda.get_device_name(0) if use_gpu else 'CPU'}")

    emb_source = gcfg.get("embeddings", "student")  # "student" or "perch"
    if emb_source == "student":
        emb_path = str(Path(cfg["outputs"]["embeddings_h5"]).parent / "student_embeddings.h5")
    else:
        emb_path = cfg["outputs"]["embeddings_h5"]
    logger.info(f"Embeddings source: {emb_source} ({emb_path})")

    segments = pd.read_csv(cfg["outputs"]["segments_csv"], low_memory=False)
    label_map = build_label_map(cfg["data"]["taxonomy_csv"])
    with h5py.File(emb_path, "r") as h5:
        written = h5["written"][:]
        D = h5["global_embeddings"].shape[1]

    rng = np.random.default_rng(gcfg.get("seed", 42))
    per_species = int(gcfg["per_species_sample"])
    sample_idx = _sample_stratified(segments, written, label_map, per_species, rng)
    logger.info(
        f"Sampled {len(sample_idx)} segments stratified across {len(label_map)} species "
        f"(≤{per_species}/species)"
    )

    with h5py.File(emb_path, "r") as h5:
        embs = h5["global_embeddings"][sample_idx]   # (N_sample, D)
    embs_nn = np.maximum(embs, 0.0).astype(np.float32)
    logger.info(f"Embeddings: shape={embs_nn.shape}, mean_nonzero_frac={(embs_nn > 0).mean():.3f}")

    # NMF expects V of shape (features, samples). Here features = D, samples = N.
    V = embs_nn.T.copy()                              # (D, N_sample)

    t0 = time.time()
    W, result = run_nmfk(
        V,
        k_min=gcfg["k_min"], k_max=gcfg["k_max"], k_step=gcfg["k_step"],
        n_runs=gcfg["n_runs_per_k"], max_iter=gcfg["nmf_max_iter"],
        perturb_std=gcfg["perturb_std"], silhouette_min=gcfg["silhouette_min"],
        seed=gcfg.get("seed", 42), algo=algo, use_gpu=use_gpu,
        selection="nmfk", strict=gcfg.get("strict", True),
    )
    logger.info(f"pyNMFk sweep finished in {(time.time() - t0)/60:.1f}m")
    logger.info("k_sweep:\n" + format_results_table(result.results))
    logger.info(
        f"Selected k={result.selected_k} via {result.selected_method} "
        f"(robustness={result.selected_robustness:+.3f}, "
        f"rel_err={result.selected_error:.4f})"
    )

    # Precompute pseudoinverse W_pinv for downstream projection
    WtW = W.T @ W
    ridge = 1e-4 * np.trace(WtW) / max(W.shape[1], 1)
    WtW += ridge * np.eye(W.shape[1], dtype=np.float32)
    W_pinv = np.linalg.solve(WtW, W.T).astype(np.float32)  # (k, D)

    np.save(out_dir / "W.npy", W)
    np.save(out_dir / "W_pinv.npy", W_pinv)
    np.save(out_dir / "sample_indices.npy", sample_idx.astype(np.int64))
    pd.DataFrame([r.__dict__ for r in result.results]).to_csv(
        out_dir / "k_sweep.csv", index=False
    )
    with open(out_dir / "build_info.json", "w") as f:
        json.dump({
            "embeddings_source": emb_source,
            "embeddings_path": emb_path,
            "D": int(D),
            "n_sample": int(len(sample_idx)),
            "selected_k": int(result.selected_k),
            "method": result.selected_method,
            "robustness": result.selected_robustness,
            "silhouette": result.selected_silhouette,
            "aic": result.selected_aic,
            "rel_error": result.selected_error,
        }, f, indent=2)
    logger.info(f"Saved → {out_dir}: W{W.shape}, W_pinv{W_pinv.shape}, k_sweep.csv, build_info.json")


def cmd_project(cfg: dict):
    """Project all segments' embeddings through W_pinv → H (k, N_all)."""
    logger = setup_logging("nmf_global", cfg["outputs"]["logs_dir"])
    gcfg = cfg["stage_nmf_global"]
    out_dir = Path(cfg["outputs"]["nmf_global_dir"])

    W = np.load(out_dir / "W.npy")
    W_pinv = np.load(out_dir / "W_pinv.npy")
    with open(out_dir / "build_info.json") as f:
        info = json.load(f)
    emb_path = info["embeddings_path"]
    k = W.shape[1]
    logger.info(f"W{W.shape}, W_pinv{W_pinv.shape} (k={k}); projecting from {emb_path}")

    segments = pd.read_csv(cfg["outputs"]["segments_csv"], low_memory=False)
    N = len(segments)
    with h5py.File(emb_path, "r") as h5:
        written = h5["written"][:]
        embs_ds = h5["global_embeddings"]
        H_all = np.zeros((N, k), dtype=np.float32)
        valid_idx = np.where(written)[0]
        logger.info(f"Projecting {len(valid_idx)}/{N} valid segments")
        batch = int(gcfg.get("project_batch_size", 4096))
        t0 = time.time()
        for start in range(0, len(valid_idx), batch):
            end = min(start + batch, len(valid_idx))
            idxs = valid_idx[start:end]
            V_chunk = np.maximum(embs_ds[np.sort(idxs)], 0.0).astype(np.float32)  # (B, D)
            H_chunk = V_chunk @ W_pinv.T                                          # (B, k)
            H_all[np.sort(idxs)] = H_chunk
            if (start // batch) % 20 == 0:
                elapsed = time.time() - t0
                logger.info(
                    f"  {end}/{len(valid_idx)}  ({elapsed/60:.1f}m, "
                    f"{end / max(elapsed, 1e-6):.0f}/s)"
                )

    out_path = out_dir / "H_all.npy"
    np.save(out_path, H_all)
    logger.info(f"Saved H_all {H_all.shape} → {out_path}")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", choices=["build", "project", "all"])
    args, _ = parser.parse_known_args()
    cfg = get_config()
    if args.command == "build":
        cmd_build(cfg)
    elif args.command == "project":
        cmd_project(cfg)
    else:
        cmd_build(cfg)
        cmd_project(cfg)


if __name__ == "__main__":
    main()
