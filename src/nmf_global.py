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
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch

from config import get_config
from utils import setup_logging, build_label_map


_PYNMFK_SRC = Path.home() / "pyNMFk" / "src"
if _PYNMFK_SRC.exists() and str(_PYNMFK_SRC) not in sys.path:
    sys.path.insert(0, str(_PYNMFK_SRC))

from pynmfk.solvers import run_nmf as pynmfk_run_nmf  # noqa: E402
from pynmfk.selection import (  # noqa: E402
    compute_cluster_silhouettes,
    compute_aic,
)


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


def _run_nmf_once(V: np.ndarray, k: int, max_iter: int,
                  seed: int, perturb_std: float,
                  algo: str = "hals", use_gpu: bool = True):
    """Single perturbed NMF fit. Returns (W, H, rel_error_vs_clean_V)."""
    rng = np.random.RandomState(seed)
    noise = perturb_std * V.mean() * rng.randn(*V.shape).astype(np.float32)
    V_pert = np.maximum(V + noise, 1e-10).astype(np.float32)

    with contextlib.redirect_stdout(io.StringIO()):
        W, H, _ = pynmfk_run_nmf(
            V_pert, k=k, algo=algo, max_iter=max_iter, seed=seed, use_gpu=use_gpu,
        )
    V_norm = float(np.linalg.norm(V, "fro")) + 1e-10
    err = float(np.linalg.norm(V - W @ H, "fro") / V_norm)
    return W.astype(np.float32), H.astype(np.float32), err


def _select_k_nmfk(V: np.ndarray, k_min: int, k_max: int, k_step: int,
                    n_runs: int, max_iter: int, perturb_std: float,
                    cutoff: float, seed: int,
                    algo: str = "hals", use_gpu: bool = True,
                    strict: bool = True, logger=None):
    """Canonical NMFk.jl selection: largest k with robustness > cutoff.

    Returns (best_k, best_W, diagnostics).
    """
    f, T = V.shape
    k_max = min(k_max, f - 1, T - 1)
    ks = list(range(k_min, k_max + 1, k_step))
    if logger:
        logger.info(f"Sweep k ∈ {ks}, {n_runs} runs each")

    results = []
    best_W_per_k = {}
    for k in ks:
        t_k = time.time()
        W_runs, errs, best_err, best_W, best_H = [], [], float("inf"), None, None
        for r in range(n_runs):
            try:
                W, H, err = _run_nmf_once(
                    V, k, max_iter=max_iter,
                    seed=seed + r, perturb_std=perturb_std,
                    algo=algo, use_gpu=use_gpu,
                )
            except Exception as e:
                if logger:
                    logger.warning(f"  k={k} run={r}: {e}")
                continue
            W_runs.append(W)
            errs.append(err)
            if err < best_err:
                best_err, best_W, best_H = err, W, H
        if not W_runs or best_W is None:
            continue
        cluster_sils = compute_cluster_silhouettes(W_runs)
        robustness = float(cluster_sils.min())
        mean_sil = float(cluster_sils.mean())
        aic = compute_aic(V, best_W, best_H, k)
        results.append({
            "k": k, "robustness": robustness, "mean_silhouette": mean_sil,
            "mean_rel_error": float(np.mean(errs)),
            "std_rel_error": float(np.std(errs)),
            "aic": float(aic),
        })
        best_W_per_k[k] = best_W
        if logger:
            logger.info(
                f"  k={k:3d}  rob={robustness:+.3f}  sil_mean={mean_sil:+.3f}  "
                f"rel_err={np.mean(errs):.4f}  aic={aic:.1f}  "
                f"t={time.time() - t_k:.1f}s"
            )

    if not results:
        raise RuntimeError("No successful NMF fits across candidate ks")

    passing = [r for r in results if r["robustness"] > cutoff]
    if passing:
        best = max(passing, key=lambda r: r["k"])
        method = "nmfk_largest_stable"
    elif not strict:
        best = max(results, key=lambda r: r["robustness"])
        method = "nmfk_max_robustness_fallback"
    else:
        best = max(results, key=lambda r: r["robustness"])
        method = "nmfk_degraded"
    return best["k"], best_W_per_k[best["k"]], {
        "results": results, "selected": best, "method": method,
    }


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

    best_k, W, diag = _select_k_nmfk(
        V,
        k_min=gcfg["k_min"], k_max=gcfg["k_max"], k_step=gcfg["k_step"],
        n_runs=gcfg["n_runs_per_k"], max_iter=gcfg["nmf_max_iter"],
        perturb_std=gcfg["perturb_std"], cutoff=gcfg["silhouette_min"],
        seed=gcfg.get("seed", 42), algo=algo, use_gpu=use_gpu,
        strict=gcfg.get("strict", True), logger=logger,
    )
    logger.info(
        f"Selected k={best_k} via {diag['method']} "
        f"(robustness={diag['selected']['robustness']:+.3f}, "
        f"rel_err={diag['selected']['mean_rel_error']:.4f})"
    )

    # Precompute pseudoinverse W_pinv for downstream projection
    WtW = W.T @ W
    ridge = 1e-4 * np.trace(WtW) / max(W.shape[1], 1)
    WtW += ridge * np.eye(W.shape[1], dtype=np.float32)
    W_pinv = np.linalg.solve(WtW, W.T).astype(np.float32)  # (k, D)

    np.save(out_dir / "W.npy", W)
    np.save(out_dir / "W_pinv.npy", W_pinv)
    np.save(out_dir / "sample_indices.npy", sample_idx.astype(np.int64))
    pd.DataFrame(diag["results"]).to_csv(out_dir / "k_sweep.csv", index=False)
    with open(out_dir / "build_info.json", "w") as f:
        json.dump({
            "embeddings_source": emb_source,
            "embeddings_path": emb_path,
            "D": int(D),
            "n_sample": int(len(sample_idx)),
            "selected_k": int(best_k),
            "method": diag["method"],
            "selected": diag["selected"],
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
