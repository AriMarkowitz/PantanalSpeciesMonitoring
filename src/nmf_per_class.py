"""Per-species NMFk dictionaries + pseudoinverse projection for feature extraction.

Pipeline:
  1. build-dicts: for each species, gather its strong_primary mel spectrograms
     from the cached mels memmap, concatenate into a matrix V_sp, run NMFk
     (rank selection by silhouette stability) to learn W_sp of shape
     (n_mels, k_sp). k_sp is species-specific.
  2. project: each species' W_sp gets its own pseudoinverse W_sp_pinv
     computed independently (block-diagonal assembly — no inter-species
     competition). Blocks are stacked into W_all_pinv of shape
     (sum_k, n_mels), so H = W_all_pinv @ V still works at inference but
     each row now depends only on its species' atoms. Per species extract
     4 scalars:
       - recon_error:  ||V_clip - W_sp @ H_sp||^2 / ||V_clip||^2
       - neg_frac:     fraction of H_sp entries below 0 (pseudoinverse only)
       - energy:       ||H_sp||^2
       - pos_energy:   ||max(H_sp, 0)||^2
     Yields 4 * num_species features per segment.

Usage:
    python src/nmf_per_class.py build-dicts
    python src/nmf_per_class.py project
    python src/nmf_per_class.py all
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch

from config import get_config
from utils import setup_logging, build_label_map
from cache_mels import get_mel_shape

# pyNMFk lives as a bare source tree at ~/pyNMFk/src/pynmfk. Add to sys.path if
# not already installed as a package (mirrors BirdCallClassifier's approach).
_PYNMFK_SRC = Path.home() / "pyNMFk" / "src"
if _PYNMFK_SRC.exists() and str(_PYNMFK_SRC) not in sys.path:
    sys.path.insert(0, str(_PYNMFK_SRC))

from pynmfk.solvers import run_nmf as pynmfk_run_nmf  # noqa: E402


def _gather_species_matrix(species_label: str,
                            segments: pd.DataFrame,
                            mels_mmap: np.memmap,
                            written: np.ndarray,
                            max_clips: int,
                            min_clips: int,
                            rng: np.random.Generator) -> np.ndarray | None:
    """Concatenate mel spectrograms of strong_primary segments for one species.

    Returns V of shape (n_mels, n_clips * T_frames), or None if too few clips.
    """
    mask = (
        (segments["label_quality"].values == "strong_primary")
        & (segments["primary_label"].astype(str).values == species_label)
    )
    idxs = np.where(mask)[0]
    # Filter to segments that were actually embedded (written==True means
    # the segment was valid during feature extraction)
    idxs = idxs[written[idxs]]
    if len(idxs) < min_clips:
        return None
    if len(idxs) > max_clips:
        idxs = rng.choice(idxs, size=max_clips, replace=False)

    mats = mels_mmap[np.sort(idxs)].astype(np.float32)  # (N_clips, n_mels, T)
    # Mel memmap is float16 normalized to [0,1]; NMF needs strictly non-negative.
    mats = np.maximum(mats, 0.0)
    n_clips, n_mels, T = mats.shape
    V = mats.transpose(1, 0, 2).reshape(n_mels, n_clips * T)
    return V


def _nmfk_cluster_silhouettes(W_runs: list[np.ndarray]) -> np.ndarray:
    """NMFk.jl-style per-cluster silhouettes (cosine on component vectors).

    For each k-value run, components are aligned to the first run's
    components via Hungarian matching on cosine similarity, then the k
    resulting clusters of size ``n_runs`` are silhouette-scored separately.

    Returns an array of length k — one silhouette per cluster.
    NMFk.jl uses robustness = min(cluster_silhouettes) to enforce that
    ALL components are stable, not just the average.
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import silhouette_samples

    if len(W_runs) < 2:
        return np.ones(W_runs[0].shape[1], dtype=np.float64)
    k = W_runs[0].shape[1]
    W_ref = W_runs[0]
    W_ref_n = W_ref / (np.linalg.norm(W_ref, axis=0, keepdims=True) + 1e-10)

    aligned = [[] for _ in range(k)]
    for W in W_runs:
        Wn = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-10)
        cos_sim = W_ref_n.T @ Wn
        row_ind, col_ind = linear_sum_assignment(-cos_sim)
        for ref_j, run_j in zip(row_ind, col_ind):
            aligned[ref_j].append(Wn[:, run_j])

    vectors, labels = [], []
    for j in range(k):
        for vec in aligned[j]:
            vectors.append(vec)
            labels.append(j)
    X = np.array(vectors)
    y = np.array(labels)
    if len(np.unique(y)) < 2 or len(X) <= k:
        return np.ones(k, dtype=np.float64)

    # Per-sample silhouettes, then average within each cluster
    s = silhouette_samples(X, y, metric="cosine")
    per_cluster = np.zeros(k, dtype=np.float64)
    for j in range(k):
        sj = s[y == j]
        per_cluster[j] = float(sj.mean()) if sj.size > 0 else 1.0
    return per_cluster


def _run_nmf_once(V: np.ndarray, k: int, max_iter: int,
                  seed: int, perturb_std: float,
                  algo: str = "hals",
                  use_gpu: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """Run a single NMF fit with bootstrap perturbation (GPU via pyNMFk).

    Returns W (f, k), H (k, T), relative_error.
    """
    rng = np.random.RandomState(seed)
    noise = perturb_std * V.mean() * rng.randn(*V.shape).astype(np.float32)
    V_pert = np.maximum(V + noise, 1e-10).astype(np.float32)

    # nmf-torch prints "Use GPU mode." + per-10-iter loss lines. Across
    # 234 species × ~6 k-values × 3 runs that's ~60k lines of noise. Swallow.
    with contextlib.redirect_stdout(io.StringIO()):
        W, H, rel_err_pert = pynmfk_run_nmf(
            V_pert, k=k, algo=algo, max_iter=max_iter, seed=seed, use_gpu=use_gpu,
        )
    # Report reconstruction error against the ORIGINAL (unperturbed) V
    V_norm = float(np.linalg.norm(V, "fro")) + 1e-10
    err = float(np.linalg.norm(V - W @ H, "fro") / V_norm)
    return W.astype(np.float32), H.astype(np.float32), err


def _select_k_for_species(V: np.ndarray, k_min: int, k_max: int, k_step: int,
                           n_runs: int, max_iter: int, perturb_std: float,
                           silhouette_min: float, seed: int,
                           algo: str = "hals",
                           use_gpu: bool = True,
                           selection: str = "nmfk",
                           strict: bool = True) -> tuple[int, np.ndarray, dict]:
    """NMFk rank selection for one species. Returns (best_k, best_W, diagnostics).

    selection:
      "nmfk"       — canonical NMFk.jl rule: robustness = min(cluster
                     silhouettes); pick the LARGEST k whose robustness exceeds
                     silhouette_min (the cutoff). If strict=False and none
                     pass, fall back to max-robustness k. This is the
                     published algorithm — prefers the most detailed
                     factorization that is stable across restarts.
      "rel_error"  — lowest mean_rel_error among stable candidates (legacy;
                     trivially picks k_max because error decreases with k).
      "silhouette" — highest mean silhouette, tie-break by lower error
                     (legacy; biased toward small k).
    """
    # Clamp k_max to what the matrix can support (k ≤ min(f, T))
    f, T = V.shape
    k_max = min(k_max, f - 1, T - 1)
    if k_max < k_min:
        k_max = max(k_min, 2)
    ks = list(range(k_min, k_max + 1, k_step))

    results = []
    best_W_per_k = {}
    for k in ks:
        W_runs = []
        H_runs = []
        errs = []
        for r in range(n_runs):
            try:
                W, H, err = _run_nmf_once(V, k, max_iter=max_iter,
                                          seed=seed + r, perturb_std=perturb_std,
                                          algo=algo, use_gpu=use_gpu)
            except Exception:
                continue
            W_runs.append(W)
            H_runs.append(H)
            errs.append(err)
        if not W_runs:
            continue
        # Per-cluster silhouettes; NMFk robustness = min across clusters
        cluster_sils = _nmfk_cluster_silhouettes(W_runs)
        robustness = float(cluster_sils.min())
        mean_sil = float(cluster_sils.mean())
        best_run = int(np.argmin(errs))
        best_W_per_k[k] = W_runs[best_run]
        results.append({
            "k": k,
            "robustness": robustness,      # min cluster silhouette (NMFk.jl rule)
            "mean_silhouette": mean_sil,   # for diagnostics
            "mean_rel_error": float(np.mean(errs)),
            "std_rel_error": float(np.std(errs)),
        })

    if not results:
        # Fallback: single tiny k, single run
        W, _, _ = _run_nmf_once(V, k=min(k_min, max(2, min(f, T) - 1)),
                                max_iter=max_iter, seed=seed,
                                perturb_std=perturb_std,
                                algo=algo, use_gpu=use_gpu)
        return W.shape[1], W, {"fallback": True, "results": []}

    if selection == "nmfk":
        # NMFk.jl rule: largest k with robustness > cutoff. If none pass
        # and strict=False, pick the k with maximum robustness.
        passing = [r for r in results if r["robustness"] > silhouette_min]
        if passing:
            best = max(passing, key=lambda r: r["k"])
        elif not strict:
            best = max(results, key=lambda r: r["robustness"])
        else:
            # strict=True + nothing passes → pick max-robustness as a sane
            # default rather than returning None (the per-species loop
            # needs some W to stack). Log this as a degraded case.
            best = max(results, key=lambda r: r["robustness"])
            best["degraded"] = True
    elif selection == "silhouette":
        stable = [r for r in results if r["robustness"] >= silhouette_min]
        if not stable:
            stable = results
        best = max(stable, key=lambda r: (r["mean_silhouette"], -r["mean_rel_error"]))
    else:  # "rel_error": legacy — lowest error among stable
        stable = [r for r in results if r["robustness"] >= silhouette_min]
        if not stable:
            stable = results
        best = min(stable, key=lambda r: (r["mean_rel_error"], -r["robustness"]))
    best_k = best["k"]
    return best_k, best_W_per_k[best_k], {"results": results, "selected": best}


def cmd_build_dicts(cfg: dict):
    """Learn per-species W_sp dictionaries from strong_primary mel spectrograms."""
    logger = setup_logging("nmf_per_class", cfg["outputs"]["logs_dir"])
    pcfg = cfg["stage_nmf_pc"]
    out_dir = Path(cfg["outputs"]["nmf_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = torch.cuda.is_available()
    algo = pcfg.get("algo", "hals")
    if use_gpu:
        logger.info(f"NMF solver: pyNMFk ({algo}) on GPU {torch.cuda.get_device_name(0)}")
    else:
        logger.info(f"NMF solver: pyNMFk ({algo}) on CPU — no CUDA available")

    # Load segments + label map
    segments = pd.read_csv(cfg["outputs"]["segments_csv"], low_memory=False)
    label_map = build_label_map(cfg["data"]["taxonomy_csv"])
    num_species = len(label_map)
    logger.info(f"Species: {num_species}, Segments: {len(segments)}")

    # Open mels memmap + written flag
    emb_h5 = cfg["outputs"]["embeddings_h5"]
    mels_path = emb_h5 + ".mels.npy"
    if not Path(mels_path).exists():
        raise FileNotFoundError(
            f"Cached mels not found at {mels_path}. "
            f"Run scripts/cache_mels.sh first."
        )
    with h5py.File(emb_h5, "r") as h5:
        written = h5["written"][:]

    n_mels, T = get_mel_shape(cfg)
    N = len(segments)
    logger.info(f"Mel shape per segment: ({n_mels}, {T})")
    mels = np.memmap(mels_path, dtype=np.float16, mode="r",
                     shape=(N, n_mels, T))

    rng = np.random.default_rng(pcfg.get("seed", 42))

    # Loop species
    species_info = []  # one row per species with k_sp, n_clips, diagnostics
    W_per_species = {}  # species_label → W_sp
    t0 = time.time()

    # Also build a pooled "background" matrix from species that fall below
    # the min_clips threshold — used as a shared fallback dictionary.
    fallback_pool = []

    species_list = sorted(label_map.keys())
    logger.info(
        f"Fitting {num_species} species (k range {pcfg['k_min']}..{pcfg['k_max']} "
        f"step {pcfg['k_step']}, {pcfg['n_runs_per_k']} runs/k, "
        f"selection={pcfg.get('k_selection', 'nmfk')}, "
        f"cutoff={pcfg['silhouette_min']}, strict={pcfg.get('k_strict', True)})"
    )
    n_fallback = 0
    for i, sp in enumerate(species_list):
        t_sp = time.time()
        V = _gather_species_matrix(
            sp, segments, mels, written,
            max_clips=pcfg["max_clips_per_species"],
            min_clips=pcfg["min_clips_per_species"],
            rng=rng,
        )
        if V is None:
            # Too few clips — try to collect what we can for fallback pool
            mask = (
                (segments["label_quality"].values == "strong_primary")
                & (segments["primary_label"].astype(str).values == sp)
            )
            idxs = np.where(mask)[0]
            idxs = idxs[written[idxs]]
            if len(idxs) > 0:
                mats = mels[np.sort(idxs)].astype(np.float32)
                mats = np.maximum(mats, 0.0)
                fallback_pool.append(mats.transpose(1, 0, 2).reshape(n_mels, -1))
            species_info.append({
                "species": sp, "k_sp": 0, "n_clips": int(len(idxs)),
                "fallback": True,
            })
            n_fallback += 1
            logger.info(
                f"[{i+1:3d}/{num_species}] {sp:<12s} FALLBACK  clips={len(idxs):<3d}  "
                f"(pooled → shared W)"
            )
            continue

        best_k, W_sp, diag = _select_k_for_species(
            V, k_min=pcfg["k_min"], k_max=pcfg["k_max"],
            k_step=pcfg["k_step"], n_runs=pcfg["n_runs_per_k"],
            max_iter=pcfg["nmf_max_iter"],
            perturb_std=pcfg["perturb_std"],
            silhouette_min=pcfg["silhouette_min"],
            seed=pcfg.get("seed", 42) + i,
            algo=algo, use_gpu=use_gpu,
            selection=pcfg.get("k_selection", "nmfk"),
            strict=pcfg.get("k_strict", True),
        )
        W_per_species[sp] = W_sp
        sel = diag.get("selected", {}) or {}
        rel_err = sel.get("mean_rel_error", float("nan"))
        robustness = sel.get("robustness", float("nan"))
        mean_sil = sel.get("mean_silhouette", float("nan"))
        degraded = sel.get("degraded", False)
        species_info.append({
            "species": sp, "k_sp": best_k,
            "n_clips": int(V.shape[1] // T),
            "n_frames": int(V.shape[1]),
            "robustness": robustness,
            "mean_silhouette": mean_sil,
            "rel_error": rel_err,
            "degraded": degraded,
            "fallback": False,
        })

        t_species = time.time() - t_sp
        elapsed = time.time() - t0
        done = i + 1
        remaining = num_species - done
        eta_min = (elapsed / done) * remaining / 60.0
        tag = " [DEGRADED]" if degraded else ""
        logger.info(
            f"[{done:3d}/{num_species}] {sp:<12s} k={best_k:<3d} clips={V.shape[1] // T:<3d}  "
            f"rob={robustness:+.2f} sil_mean={mean_sil:+.2f} rel_err={rel_err:.3f}  "
            f"t={t_species:5.1f}s  elapsed={elapsed/60:5.1f}m  eta={eta_min:5.1f}m{tag}"
        )

    logger.info(
        f"Per-species fitting done. real_dicts={num_species - n_fallback}, "
        f"fallback={n_fallback}, total_time={(time.time() - t0)/60:.1f}m"
    )

    # Build shared fallback W from pooled low-data species
    W_fallback = None
    if fallback_pool:
        V_pool = np.concatenate(fallback_pool, axis=1)
        k_fb = min(pcfg["k_max"], V_pool.shape[1] - 1, n_mels - 1)
        if k_fb >= 2:
            logger.info(f"Fitting fallback W on {V_pool.shape[1]} pooled frames, k={k_fb}")
            W_fallback, _, _ = _run_nmf_once(V_pool, k=k_fb,
                                             max_iter=pcfg["nmf_max_iter"],
                                             seed=pcfg.get("seed", 42),
                                             perturb_std=pcfg["perturb_std"],
                                             algo=algo, use_gpu=use_gpu)

    # Assign fallback W to species that lack their own
    for row in species_info:
        if row["fallback"] and W_fallback is not None:
            W_per_species[row["species"]] = W_fallback
            row["k_sp"] = W_fallback.shape[1]
        elif row["fallback"] and W_fallback is None:
            # No fallback → give a zeros W of minimum k so slicing stays consistent.
            # Pinv of zeros is zeros → features for this species will be zero.
            W_per_species[row["species"]] = np.zeros((n_mels, 2), dtype=np.float32)
            row["k_sp"] = 2

    # Stack into one big W_all: (n_mels, sum_k) in canonical species order
    W_blocks = [W_per_species[sp] for sp in species_list]
    boundaries = np.cumsum([0] + [W.shape[1] for W in W_blocks])  # length num_species+1
    W_all = np.concatenate(W_blocks, axis=1).astype(np.float32)
    logger.info(f"W_all shape: {W_all.shape}, total k = {boundaries[-1]}")

    # Per-species pseudoinverse: compute W_sp_pinv independently per species
    # and stack. This removes the inter-species competition that the joint
    # ridge-solve introduces: each species' H_sp is now the least-squares fit
    # against ONLY its own dictionary, so a strong fit by species A can't
    # suppress activation on species B. At inference the layout and shape
    # are identical to the joint version — H = W_all_pinv @ V still works.
    pinv_blocks = []
    for W_sp in W_blocks:
        k_sp = W_sp.shape[1]
        WtW = W_sp.T @ W_sp  # (k_sp, k_sp)
        ridge = 1e-4 * np.trace(WtW) / max(k_sp, 1)
        WtW += ridge * np.eye(k_sp, dtype=np.float32)
        W_sp_pinv = np.linalg.solve(WtW, W_sp.T).astype(np.float32)  # (k_sp, n_mels)
        pinv_blocks.append(W_sp_pinv)
    W_all_pinv = np.concatenate(pinv_blocks, axis=0).astype(np.float32)  # (sum_k, n_mels)
    logger.info(f"W_all_pinv shape: {W_all_pinv.shape} (per-species pinv blocks)")

    # Save
    np.save(out_dir / "W_all.npy", W_all)
    np.save(out_dir / "W_all_pinv.npy", W_all_pinv)
    np.save(out_dir / "species_boundaries.npy", boundaries.astype(np.int32))
    with open(out_dir / "species_order.json", "w") as f:
        json.dump(species_list, f)
    pd.DataFrame(species_info).to_csv(out_dir / "species_info.csv", index=False)

    logger.info(f"Saved dictionaries to {out_dir}")
    logger.info(f"  W_all.npy               {W_all.shape}")
    logger.info(f"  W_all_pinv.npy          {W_all_pinv.shape}")
    logger.info(f"  species_boundaries.npy  {boundaries.shape}")
    logger.info(f"  species_order.json      {len(species_list)} species")
    logger.info(f"  species_info.csv        k_sp per species")


def project_segments_to_features(
    mels_chunk: np.ndarray,
    W_all: np.ndarray,
    W_all_pinv: np.ndarray,
    boundaries: np.ndarray,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Compute per-species reconstruction + activation features for a chunk.

    Uses pseudoinverse projection: H_all = W_all_pinv @ V  (one matmul).
    Per species, extracts 4 scalar features:
        recon_error   — ||V - W_sp H_sp||^2 / ||V||^2        (lower = species fits well)
        neg_frac      — fraction of H_sp entries below 0      (pseudoinverse only)
        energy        — total ||H_sp||^2                      (how much this dictionary activates)
        pos_energy    — ||max(H_sp, 0)||^2                    (only the NMF-valid, positive part)

    The last two are separated because pseudoinverse H can be negative (no
    non-negativity constraint); the fraction-of-negative and positive-only
    energy give an extra read on whether V actually belongs in this basis.

    Args:
        mels_chunk:   (B, n_mels, T) float32 — non-negative mel spectrograms
        W_all:        (n_mels, sum_k) — stacked per-species dictionaries
        W_all_pinv:   (sum_k, n_mels) — precomputed pseudoinverse of W_all
        boundaries:   (num_species + 1,) int — column slices into W_all
        device:       "cpu" or "cuda" — where to run the matmul

    Returns:
        (B, 4 * num_species) float32 features.
        Layout: [recon_err_0..S-1, neg_frac_0..S-1, energy_0..S-1, pos_energy_0..S-1]
    """
    dev = torch.device(device)
    V = torch.from_numpy(np.asarray(mels_chunk, dtype=np.float32)).to(dev)  # (B, f, T)
    W_all_t = torch.from_numpy(np.asarray(W_all, dtype=np.float32)).to(dev)
    W_pinv_t = torch.from_numpy(np.asarray(W_all_pinv, dtype=np.float32)).to(dev)
    B, n_mels, T = V.shape
    num_species = len(boundaries) - 1

    # One big pinv matmul: H_flat = W_all_pinv @ V_flat
    # Reshape V to (f, B*T), multiply, reshape to (B, sum_k, T)
    V_flat = V.permute(1, 0, 2).reshape(n_mels, B * T)
    H_flat = W_pinv_t @ V_flat                         # (sum_k, B*T)
    sum_k = H_flat.shape[0]
    H_all = H_flat.reshape(sum_k, B, T).permute(1, 0, 2)  # (B, sum_k, T)

    V_sq = (V ** 2).sum(dim=(1, 2))                    # (B,)
    V_sq_safe = torch.clamp(V_sq, min=1e-10)

    recon_err   = torch.zeros((B, num_species), device=dev)
    neg_frac    = torch.zeros((B, num_species), device=dev)
    energy      = torch.zeros((B, num_species), device=dev)
    pos_energy  = torch.zeros((B, num_species), device=dev)

    for sp_idx in range(num_species):
        c0, c1 = int(boundaries[sp_idx]), int(boundaries[sp_idx + 1])
        if c1 <= c0:
            continue
        W_sp = W_all_t[:, c0:c1]                       # (f, k_sp)
        H_sp = H_all[:, c0:c1, :]                      # (B, k_sp, T)

        # Reconstruction V_hat = W_sp @ H_sp  per segment
        V_hat = torch.einsum("fk,bkt->bft", W_sp, H_sp)
        diff = V - V_hat
        err_sq = (diff ** 2).sum(dim=(1, 2))
        recon_err[:, sp_idx] = err_sq / V_sq_safe

        k_sp_T = H_sp.shape[1] * H_sp.shape[2]
        neg_frac[:, sp_idx] = (H_sp < 0).float().sum(dim=(1, 2)) / max(k_sp_T, 1)
        energy[:, sp_idx] = (H_sp ** 2).sum(dim=(1, 2))
        pos_energy[:, sp_idx] = (torch.clamp(H_sp, min=0.0) ** 2).sum(dim=(1, 2))

    out = torch.cat([recon_err, neg_frac, energy, pos_energy], dim=1)
    return out.detach().cpu().numpy().astype(np.float32)


def cmd_project(cfg: dict, source: str = "primary"):
    """Project all segments through per-species dictionaries → feature matrix.

    source: "primary" → outputs/segments.csv + outputs/embeddings/embeddings.h5
            "distill" → outputs/distill_segments.csv + distill_embeddings.h5
    """
    logger = setup_logging("nmf_per_class", cfg["outputs"]["logs_dir"])
    pcfg = cfg["stage_nmf_pc"]
    out_dir = Path(cfg["outputs"]["nmf_dir"])

    # Load dictionaries
    W_all = np.load(out_dir / "W_all.npy")
    W_all_pinv = np.load(out_dir / "W_all_pinv.npy")
    boundaries = np.load(out_dir / "species_boundaries.npy")
    num_species = len(boundaries) - 1
    feat_dim = 4 * num_species  # recon_err + neg_frac + energy + pos_energy per species
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"W_all {W_all.shape}, W_all_pinv {W_all_pinv.shape}, "
                f"{num_species} species → {feat_dim} features per segment "
                f"(device={device})")

    # Pick source
    if source == "distill":
        seg_csv = cfg["outputs"]["distill_segments_csv"]
        emb_h5 = cfg["outputs"]["distill_embeddings_h5"]
        out_file = out_dir / "nmf_features_distill.npy"
    else:
        seg_csv = cfg["outputs"]["segments_csv"]
        emb_h5 = cfg["outputs"]["embeddings_h5"]
        out_file = out_dir / "nmf_features.npy"

    segments = pd.read_csv(seg_csv, low_memory=False)
    N = len(segments)
    mels_path = emb_h5 + ".mels.npy"
    if not Path(mels_path).exists():
        raise FileNotFoundError(f"Cached mels not found at {mels_path}")

    n_mels, T = get_mel_shape(cfg)
    mels = np.memmap(mels_path, dtype=np.float16, mode="r",
                     shape=(N, n_mels, T))
    with h5py.File(emb_h5, "r") as h5:
        written = h5["written"][:]

    features = np.zeros((N, feat_dim), dtype=np.float32)
    batch_size = int(pcfg["project_batch_size"])
    valid_idx = np.where(written)[0]
    logger.info(f"Projecting {len(valid_idx)}/{N} valid segments, "
                f"batch_size={batch_size}")

    t0 = time.time()
    for start in range(0, len(valid_idx), batch_size):
        end = min(start + batch_size, len(valid_idx))
        idxs = valid_idx[start:end]
        chunk = mels[np.sort(idxs)].astype(np.float32)
        chunk = np.maximum(chunk, 0.0)
        feats = project_segments_to_features(
            chunk, W_all, W_all_pinv, boundaries, device=device,
        )
        features[np.sort(idxs)] = feats
        if (start // batch_size) % 20 == 0:
            elapsed = time.time() - t0
            logger.info(f"  {end}/{len(valid_idx)} segments "
                        f"({elapsed/60:.1f}min, "
                        f"{end / max(elapsed, 1e-6):.0f} segments/s)")

    np.save(out_file, features)
    logger.info(f"Saved {features.shape} → {out_file}")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", choices=["build-dicts", "project", "all"],
                        help="Which stage to run")
    parser.add_argument("--source", choices=["primary", "distill"],
                        default="primary",
                        help="Which segments CSV to project (default: primary)")
    args, _ = parser.parse_known_args()

    cfg = get_config()

    if args.command == "build-dicts":
        cmd_build_dicts(cfg)
    elif args.command == "project":
        cmd_project(cfg, source=args.source)
    elif args.command == "all":
        cmd_build_dicts(cfg)
        cmd_project(cfg, source=args.source)


if __name__ == "__main__":
    main()
