"""Diagnose whether a global embedding-NMF dictionary is species-discriminative.

Runs four analyses on H_all = (N, k) projected from src/nmf_global.py:

  1. Motif → species heatmap: for each motif, mean activation across each
     species. Good motifs fire on few species.
  2. Per-motif max AUC: for each motif, take its best-discriminating
     species (positive = clips of that species; score = H_all[:, m]).
     Distribution tells us how many motifs carry species-specific signal.
  3. Classifier-free top-k accuracy: for each clip, use argmax motif;
     predict the species most associated with that motif. Compare to chance.
  4. Top-species-per-motif CSV: for inspection / interpretability spot checks.

All analyses run on strong_primary clips whose primary_label is a known species.

Usage:
    python src/analyze_nmf_global.py
    python src/analyze_nmf_global.py --max-clips 5000
    python src/analyze_nmf_global.py --out outputs/nmf_global/analysis/
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
from utils import build_label_map


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_scores = np.concatenate([pos, neg])
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(all_scores) + 1)
    _, inv, counts = np.unique(all_scores, return_inverse=True, return_counts=True)
    sum_ranks = np.zeros_like(counts, dtype=np.float64)
    np.add.at(sum_ranks, inv, ranks)
    ranks = (sum_ranks / counts)[inv]
    rank_pos = ranks[: len(pos)].sum()
    n_pos, n_neg = len(pos), len(neg)
    u = rank_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def analyze(cfg: dict, max_clips: int | None, seed: int, out_dir: Path | None):
    nmf_dir = Path(cfg["outputs"]["nmf_global_dir"])
    W = np.load(nmf_dir / "W.npy")                            # (D, k)
    H_all = np.load(nmf_dir / "H_all.npy")                    # (N, k)
    with open(nmf_dir / "build_info.json") as f:
        info = json.load(f)
    emb_path = info["embeddings_path"]
    k = W.shape[1]
    print(f"Loaded W{W.shape}, H_all{H_all.shape}  (k={k})")

    label_map = build_label_map(cfg["data"]["taxonomy_csv"])
    inv_label = {v: k_ for k_, v in label_map.items()}
    S = len(label_map)

    segments = pd.read_csv(cfg["outputs"]["segments_csv"], low_memory=False)
    with h5py.File(emb_path, "r") as h5:
        written = h5["written"][:]
    primary = segments["primary_label"].astype(str).values

    mask = (
        (segments["label_quality"].values == "strong_primary")
        & written
        & np.array([p in label_map for p in primary])
    )
    valid = np.where(mask)[0]
    print(f"Eligible clips: {len(valid)}")

    rng = np.random.default_rng(seed)
    if max_clips is not None and len(valid) > max_clips:
        valid = np.sort(rng.choice(valid, size=max_clips, replace=False))
        print(f"  subsampled → {len(valid)}")

    H = H_all[valid]                                          # (N_eval, k)
    y = np.array([label_map[primary[i]] for i in valid])      # species index per clip
    N_eval = len(valid)

    # ── 1. Motif → species mean activation (k, S) ────────────────────────────
    print("\n========== 1. Motif → species mean-activation heatmap ==========")
    # Aggregate mean H per species
    mean_per_species = np.zeros((S, k), dtype=np.float64)
    count_per_species = np.zeros(S, dtype=np.int64)
    for sp_idx in range(S):
        sel = (y == sp_idx)
        count_per_species[sp_idx] = int(sel.sum())
        if sel.any():
            mean_per_species[sp_idx] = H[sel].mean(axis=0)

    active_species = np.where(count_per_species > 0)[0]
    # For each motif: entropy of its activation distribution across species
    # (lower = more concentrated = more species-specific)
    mot_mean_by_sp = mean_per_species[active_species].T              # (k, S_active)
    mot_mean_by_sp_nonneg = np.maximum(mot_mean_by_sp, 0.0)
    row_sums = mot_mean_by_sp_nonneg.sum(axis=1, keepdims=True) + 1e-12
    p = mot_mean_by_sp_nonneg / row_sums                             # (k, S_active) normalized
    # Shannon entropy per motif (in bits), normalized to [0, 1]
    ent = -np.sum(p * np.log2(p + 1e-12), axis=1)
    ent_max = np.log2(len(active_species))
    ent_norm = ent / ent_max                                          # 0 = species-specific, 1 = uniform
    print(f"  k motifs: {k},  species with data: {len(active_species)}")
    print(f"  motif concentration (0 = pure-species, 1 = uniform across species):")
    print(f"    min:    {ent_norm.min():.3f}")
    print(f"    mean:   {ent_norm.mean():.3f}")
    print(f"    median: {np.median(ent_norm):.3f}")
    print(f"    max:    {ent_norm.max():.3f}")
    print(f"  fraction of motifs with ent_norm < 0.5 (concentrated): "
          f"{(ent_norm < 0.5).mean():.3f}")

    # ── 2. Per-motif best-species AUC ────────────────────────────────────────
    print("\n========== 2. Per-motif max AUC (best species per motif) ==========")
    # For each motif, find the species where it discriminates best.
    # Cheap heuristic: for each motif, the candidate species is the one with
    # highest mean activation. Compute AUC on that species.
    top_species_per_motif = mean_per_species[active_species].argmax(axis=0)
    top_species_per_motif = active_species[top_species_per_motif]    # (k,) indices into all species
    max_auc = np.full(k, np.nan, dtype=np.float64)
    for m in range(k):
        sp = int(top_species_per_motif[m])
        labels_bin = (y == sp).astype(np.int64)
        if labels_bin.sum() == 0 or labels_bin.sum() == N_eval:
            continue
        max_auc[m] = _binary_auc(H[:, m], labels_bin)
    finite = max_auc[~np.isnan(max_auc)]
    print(f"  AUC of each motif against its best species (argmax mean activation):")
    print(f"    mean:   {finite.mean():.3f}")
    print(f"    median: {np.median(finite):.3f}")
    print(f"    frac > 0.7:  {(finite > 0.7).mean():.3f}")
    print(f"    frac > 0.85: {(finite > 0.85).mean():.3f}")
    print(f"    min / max:   {finite.min():.3f} / {finite.max():.3f}")

    # ── 3. Classifier-free top-k accuracy ────────────────────────────────────
    # For each clip, predict species = species most associated with argmax motif.
    print("\n========== 3. Classifier-free top-k accuracy ==========")
    argmax_motif = H.argmax(axis=1)                                   # (N_eval,)
    predicted_species = top_species_per_motif[argmax_motif]
    top1 = (predicted_species == y).mean()
    # Top-k: pick top-k motifs per clip, aggregate candidate species
    for K in [1, 5, 10]:
        topk_motifs = np.argsort(-H, axis=1)[:, :K]                   # (N_eval, K)
        topk_species = top_species_per_motif[topk_motifs]             # (N_eval, K)
        hits = np.any(topk_species == y[:, None], axis=1).mean()
        print(f"  top-{K} (motif-vote): {hits:.4f}   (chance = K/{S} = {K/S:.4f})")
    print(f"  top-1 motif argmax → predicted species: {top1:.4f}")

    # ── 4. Save artifacts ────────────────────────────────────────────────────
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

        # Top-3 species per motif
        top3 = np.argsort(-mean_per_species[active_species], axis=0)[:3]  # (3, k)
        rows = []
        for m in range(k):
            rows.append({
                "motif": m,
                "max_auc": float(max_auc[m]) if not np.isnan(max_auc[m]) else None,
                "ent_norm": float(ent_norm[m]),
                "top1_species": inv_label[int(active_species[top3[0, m]])],
                "top2_species": inv_label[int(active_species[top3[1, m]])],
                "top3_species": inv_label[int(active_species[top3[2, m]])],
                "top1_mean_activation": float(mean_per_species[active_species[top3[0, m]], m]),
            })
        pd.DataFrame(rows).to_csv(out_dir / "motif_summary.csv", index=False)

        # Plots
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        axes[0].hist(ent_norm, bins=30)
        axes[0].axvline(0.5, color="k", lw=0.5, ls="--")
        axes[0].set_xlabel("motif species-entropy (0 = concentrated, 1 = uniform)")
        axes[0].set_ylabel("# motifs")
        axes[0].set_title(f"Motif concentration  (lower = better)")

        axes[1].hist(finite, bins=30)
        axes[1].axvline(0.5, color="k", lw=0.5, ls="--")
        axes[1].axvline(0.7, color="g", lw=0.5, ls="--")
        axes[1].set_xlabel("AUC of motif vs its best species")
        axes[1].set_ylabel("# motifs")
        axes[1].set_title(f"Per-motif max AUC  (μ={finite.mean():.3f})")

        # Heatmap of mean activation (cap to 40 species × all motifs for readability)
        # Pick species by clip count descending
        order_sp = active_species[np.argsort(-count_per_species[active_species])]
        show_sp = order_sp[: min(40, len(order_sp))]
        heat = mean_per_species[show_sp]                              # (≤40, k)
        im = axes[2].imshow(heat, aspect="auto", cmap="magma",
                            interpolation="nearest")
        axes[2].set_xlabel("motif")
        axes[2].set_ylabel("species (top-40 by count)")
        axes[2].set_title("Mean activation (species × motif)")
        fig.colorbar(im, ax=axes[2], fraction=0.02)

        fig.tight_layout()
        fig.savefig(out_dir / "nmf_global_discriminability.png",
                    dpi=150, bbox_inches="tight")

        summary = {
            "n_clips_evaluated": int(N_eval),
            "k_motifs": int(k),
            "n_active_species": int(len(active_species)),
            "motif_ent_norm": {
                "mean": float(ent_norm.mean()),
                "median": float(np.median(ent_norm)),
                "frac_concentrated_lt_0_5": float((ent_norm < 0.5).mean()),
            },
            "motif_max_auc": {
                "mean": float(finite.mean()),
                "median": float(np.median(finite)),
                "frac_gt_0_7": float((finite > 0.7).mean()),
                "frac_gt_0_85": float((finite > 0.85).mean()),
            },
            "chance_top1": 1.0 / S,
            "motif_vote_top1": float(top1),
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved → {out_dir}/  (motif_summary.csv, nmf_global_discriminability.png, summary.json)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="outputs/nmf_global/analysis")
    args = parser.parse_args()

    cfg = get_config()
    out = Path(args.out) if args.out else None
    analyze(cfg, args.max_clips, args.seed, out)


if __name__ == "__main__":
    main()
