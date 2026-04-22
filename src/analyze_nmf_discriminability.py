"""Diagnose whether per-species NMF features discriminate species.

Two analyses, both run on strong_primary clips whose primary_label is in the
NMF species_order (i.e. has a real learned dictionary):

  A. Per-clip ranking — for each clip, rank all species by:
       (i)  recon_err[sp]     (lower = better fit)
       (ii) pos_energy[sp]    (higher = stronger activation)
     Headline: top-1 / top-5 / top-20 / median-rank of the TRUE species.

  B. Per-species discriminability — for each species, compare the feature
     value on clips where it IS the primary label vs where it isn't. AUC
     (larger = more discriminative). One number per species; also aggregate.

Run:
    python src/analyze_nmf_discriminability.py
    python src/analyze_nmf_discriminability.py --max-clips 2000
    python src/analyze_nmf_discriminability.py --out outputs/nmf_viz/analysis/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

from config import get_config
from cache_mels import get_mel_shape


def _per_species_stats(
    V: np.ndarray,
    W_all: np.ndarray,
    W_pinv: np.ndarray,
    boundaries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (recon_err[num_species], pos_energy[num_species]) for one clip."""
    H_all = W_pinv @ V                              # (sum_k, T)
    V_sq = float(np.sum(V ** 2)) + 1e-10
    S = len(boundaries) - 1
    recon = np.full(S, np.nan, dtype=np.float32)
    pose = np.full(S, -np.inf, dtype=np.float64)
    for sp in range(S):
        c0, c1 = int(boundaries[sp]), int(boundaries[sp + 1])
        if c1 <= c0:
            continue
        W_sp = W_all[:, c0:c1]
        H_sp = H_all[c0:c1, :]
        V_hat = W_sp @ H_sp
        recon[sp] = float(np.sum((V - V_hat) ** 2)) / V_sq
        pose[sp] = float(np.sum(np.maximum(H_sp, 0.0) ** 2))
    return recon, pose


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann–Whitney U formulation of binary AUC. Higher score → label=1.

    Returns NaN if only one class is present.
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Rank all scores together
    all_scores = np.concatenate([pos, neg])
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(all_scores) + 1)
    # Average tied ranks
    _, inv, counts = np.unique(all_scores, return_inverse=True, return_counts=True)
    # group-mean rank per value
    sum_ranks = np.zeros_like(counts, dtype=np.float64)
    np.add.at(sum_ranks, inv, ranks)
    mean_ranks = sum_ranks / counts
    ranks = mean_ranks[inv]
    rank_pos = ranks[: len(pos)].sum()
    n_pos, n_neg = len(pos), len(neg)
    u = rank_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _topk_hits(rank: np.ndarray, ks=(1, 5, 10, 20, 50)) -> dict:
    n = len(rank)
    return {f"top{k}": float((rank < k).mean()) for k in ks}


def analyze(cfg: dict, max_clips: int | None, seed: int, out_dir: Path | None):
    nmf_dir = Path(cfg["outputs"]["nmf_dir"])
    W_all = np.load(nmf_dir / "W_all.npy")
    W_pinv = np.load(nmf_dir / "W_all_pinv.npy")
    boundaries = np.load(nmf_dir / "species_boundaries.npy")
    with open(nmf_dir / "species_order.json") as f:
        species_order = json.load(f)
    sp_to_idx = {s: i for i, s in enumerate(species_order)}
    S = len(species_order)

    segments = pd.read_csv(cfg["outputs"]["segments_csv"], low_memory=False)
    with h5py.File(cfg["outputs"]["embeddings_h5"], "r") as h5:
        written = h5["written"][:]

    primary = segments["primary_label"].astype(str).values
    in_species = np.array([p in sp_to_idx for p in primary])
    mask = (segments["label_quality"].values == "strong_primary") & written & in_species
    valid = np.where(mask)[0]
    print(f"Eligible clips (strong_primary, embedded, species in NMF set): {len(valid)}")

    rng = np.random.default_rng(seed)
    if max_clips is not None and len(valid) > max_clips:
        valid = rng.choice(valid, size=max_clips, replace=False)
        valid.sort()
        print(f"  subsampled → {len(valid)}")

    n_mels, T = get_mel_shape(cfg)
    mels_path = cfg["outputs"]["embeddings_h5"] + ".mels.npy"
    mels = np.memmap(mels_path, dtype=np.float16, mode="r",
                     shape=(len(segments), n_mels, T))

    # ── Compute per-clip per-species stats ───────────────────────────────────
    N = len(valid)
    recon_mat = np.full((N, S), np.nan, dtype=np.float32)
    pose_mat = np.full((N, S), -np.inf, dtype=np.float64)
    true_idx = np.empty(N, dtype=np.int64)

    print(f"Computing per-species stats for {N} clips...")
    for i, ci in enumerate(valid):
        V = np.maximum(mels[ci].astype(np.float32), 0.0)
        recon_mat[i], pose_mat[i] = _per_species_stats(V, W_all, W_pinv, boundaries)
        true_idx[i] = sp_to_idx[primary[ci]]
        if (i + 1) % 200 == 0 or i + 1 == N:
            print(f"  {i + 1}/{N}")

    # ── Analysis A: per-clip ranking ─────────────────────────────────────────
    # For recon: lower is better → rank ascending
    # For pos_energy: higher is better → rank descending
    # Handle NaN / -inf by pushing them to the worst rank.
    recon_sort = recon_mat.copy()
    recon_sort[np.isnan(recon_sort)] = np.inf
    recon_rank_order = np.argsort(recon_sort, axis=1)    # position 0 = best
    rank_recon = np.empty(N, dtype=np.int64)
    for i in range(N):
        rank_recon[i] = int(np.where(recon_rank_order[i] == true_idx[i])[0][0])

    pose_sort = -pose_mat
    pose_sort[np.isinf(pose_sort) & (pose_sort > 0)] = np.inf  # -inf → rank last
    pose_rank_order = np.argsort(pose_sort, axis=1)
    rank_pose = np.empty(N, dtype=np.int64)
    for i in range(N):
        rank_pose[i] = int(np.where(pose_rank_order[i] == true_idx[i])[0][0])

    hits_recon = _topk_hits(rank_recon)
    hits_pose = _topk_hits(rank_pose)

    print("\n========== A. Per-clip ranking  (N = {}) ==========".format(N))
    print(f"Chance top-1:  {1 / S:.4f}   (1 / {S})")
    print("\nRanking by RECON_ERR  (argmin ⇒ pred species)")
    for k, v in hits_recon.items():
        print(f"  {k}: {v:.4f}")
    print(f"  median_rank: {int(np.median(rank_recon))}   mean_rank: {rank_recon.mean():.1f}")

    print("\nRanking by POS_ENERGY  (argmax ⇒ pred species)")
    for k, v in hits_pose.items():
        print(f"  {k}: {v:.4f}")
    print(f"  median_rank: {int(np.median(rank_pose))}   mean_rank: {rank_pose.mean():.1f}")

    # ── Analysis B: per-species AUC ──────────────────────────────────────────
    # For a species sp, positive = (true_idx == sp). Score:
    #   recon: use -recon[:,sp]  (lower err ⇒ more positive-class-like)
    #   pose:  use +pose[:,sp]
    auc_recon = np.full(S, np.nan, dtype=np.float64)
    auc_pose = np.full(S, np.nan, dtype=np.float64)
    n_pos = np.zeros(S, dtype=np.int64)
    for sp in range(S):
        y = (true_idx == sp).astype(np.int64)
        n_pos[sp] = int(y.sum())
        if n_pos[sp] == 0 or n_pos[sp] == N:
            continue
        s_recon = -recon_mat[:, sp]
        s_pose = pose_mat[:, sp]
        # Replace NaN/-inf with worst-score so they don't corrupt ranking
        s_recon = np.where(np.isnan(s_recon), -np.inf, s_recon)
        s_recon_finite = np.where(np.isinf(s_recon), np.nanmin(s_recon[np.isfinite(s_recon)]) - 1.0, s_recon) \
            if np.any(np.isinf(s_recon)) else s_recon
        s_pose_finite = np.where(np.isinf(s_pose), np.nanmin(s_pose[np.isfinite(s_pose)]) - 1.0, s_pose) \
            if np.any(np.isinf(s_pose)) else s_pose
        auc_recon[sp] = _binary_auc(s_recon_finite, y)
        auc_pose[sp] = _binary_auc(s_pose_finite, y)

    species_with_support = np.where(n_pos > 0)[0]
    print("\n========== B. Per-species AUC  ({} species with ≥1 positive clip) ==========".format(
        len(species_with_support)))
    for name, arr in [("RECON_ERR (-score)", auc_recon), ("POS_ENERGY (+score)", auc_pose)]:
        a = arr[species_with_support]
        a = a[~np.isnan(a)]
        print(f"\n  {name}")
        print(f"    mean AUC:   {a.mean():.4f}")
        print(f"    median AUC: {np.median(a):.4f}")
        print(f"    frac > 0.5: {(a > 0.5).mean():.4f}   frac > 0.7: {(a > 0.7).mean():.4f}")
        print(f"    min / max:  {a.min():.4f} / {a.max():.4f}")

    # Worst + best species by recon AUC — often the most informative
    sp_df = pd.DataFrame({
        "species": species_order,
        "n_pos": n_pos,
        "auc_recon": auc_recon,
        "auc_pose": auc_pose,
    })
    sp_df = sp_df[sp_df["n_pos"] > 0].copy()
    print("\n  Top 10 species by recon AUC:")
    print(sp_df.nlargest(10, "auc_recon").to_string(index=False))
    print("\n  Bottom 10 species by recon AUC:")
    print(sp_df.nsmallest(10, "auc_recon").to_string(index=False))

    # ── Save artifacts ───────────────────────────────────────────────────────
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        sp_df.to_csv(out_dir / "per_species_auc.csv", index=False)

        summary = {
            "n_clips": int(N),
            "n_species": int(S),
            "chance_top1": 1 / S,
            "ranking_recon": hits_recon
                | {"median_rank": int(np.median(rank_recon)),
                   "mean_rank": float(rank_recon.mean())},
            "ranking_pos_energy": hits_pose
                | {"median_rank": int(np.median(rank_pose)),
                   "mean_rank": float(rank_pose.mean())},
            "auc_recon_mean": float(np.nanmean(auc_recon)),
            "auc_pose_mean": float(np.nanmean(auc_pose)),
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        bins = np.arange(0, S + 1, max(1, S // 50))
        axes[0].hist(rank_recon, bins=bins, alpha=0.7, label="recon_err")
        axes[0].hist(rank_pose, bins=bins, alpha=0.7, label="pos_energy")
        axes[0].axvline(0.5, color="k", lw=0.5, ls="--")
        axes[0].set_title(f"True-species rank (0 = best). Chance median ≈ {S//2}")
        axes[0].set_xlabel("rank of true species")
        axes[0].set_ylabel("# clips")
        axes[0].legend()

        a_r = auc_recon[~np.isnan(auc_recon)]
        a_p = auc_pose[~np.isnan(auc_pose)]
        axes[1].hist(a_r, bins=30, alpha=0.7, label=f"recon (μ={a_r.mean():.3f})")
        axes[1].hist(a_p, bins=30, alpha=0.7, label=f"pos_E (μ={a_p.mean():.3f})")
        axes[1].axvline(0.5, color="k", lw=0.5, ls="--")
        axes[1].set_title("Per-species AUC distribution")
        axes[1].set_xlabel("AUC")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(out_dir / "discriminability.png", dpi=150, bbox_inches="tight")
        print(f"\nSaved → {out_dir}/  (per_species_auc.csv, summary.json, discriminability.png)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-clips", type=int, default=None,
                        help="subsample clips for speed (default: use all eligible)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="outputs/nmf_viz/analysis",
                        help="directory for CSV/JSON/PNG artifacts (empty to skip)")
    args = parser.parse_args()

    cfg = get_config()
    out = Path(args.out) if args.out else None
    analyze(cfg, args.max_clips, args.seed, out)


if __name__ == "__main__":
    main()
