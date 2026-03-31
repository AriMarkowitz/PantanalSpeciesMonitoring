"""Cluster-label relationship analysis.

Reads cluster_species_table.npz and species_gmm_summary.csv and prints
a summary of how well the soft clusters align with known species labels.

Usage:
    python src/analyze_clusters.py
    python src/analyze_clusters.py --top-n 20 --min-count 50
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import entropy

from config import get_config


def load_table(proto_dir: Path):
    data = np.load(proto_dir / "cluster_species_table.npz", allow_pickle=True)
    return data["counts"], data["probs"], data["species_names"]


def cluster_purity(probs: np.ndarray) -> np.ndarray:
    """1 - normalized entropy per cluster. 1=pure, 0=uniform."""
    max_ent = np.log(probs.shape[1])
    ents = np.array([entropy(p + 1e-10) for p in probs])
    return 1.0 - ents / max_ent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=15,
                        help="Number of top/bottom clusters to show")
    parser.add_argument("--min-count", type=int, default=20,
                        help="Min labeled segments in a cluster to include")
    parser.add_argument("--top-species", type=int, default=20,
                        help="Species to show in coverage table")
    args = parser.parse_args()

    cfg = get_config()
    proto_dir = Path(cfg["outputs"]["prototypes_dir"])
    gmm_path = proto_dir / "species_gmm_summary.csv"

    counts, probs, species_names = load_table(proto_dir)
    K, S = probs.shape
    total_per_cluster = counts.sum(axis=1)

    print(f"\n{'='*60}")
    print(f"CLUSTER-LABEL ANALYSIS")
    print(f"{'='*60}")
    print(f"Global clusters: {K}  |  Species: {S}")
    print(f"Total labeled segment-cluster assignments: {total_per_cluster.sum():.0f}")

    # ── 1. Overall purity distribution ───────────────────────────────────────
    active = total_per_cluster >= args.min_count
    purity = cluster_purity(probs)
    p_active = purity[active]

    print(f"\n── Cluster Purity (min_count={args.min_count}, n={active.sum()}) ──")
    for thresh, label in [(0.9, ">0.9 (species-pure)"),
                          (0.7, ">0.7 (mostly pure)"),
                          (0.5, ">0.5 (majority)"),
                          (0.3, ">0.3 (weak signal)"),
                          (0.0, "≤0.3 (noise/mixed)")]:
        if thresh == 0.0:
            n = (p_active <= 0.3).sum()
        else:
            n = (p_active > thresh).sum()
        pct = 100 * n / len(p_active) if len(p_active) else 0
        print(f"  {label:35s}: {n:4d}  ({pct:.1f}%)")

    print(f"\n  Mean purity:   {p_active.mean():.3f}")
    print(f"  Median purity: {np.median(p_active):.3f}")

    # ── 2. Most species-pure clusters ────────────────────────────────────────
    idx_active = np.where(active)[0]
    sorted_by_purity = idx_active[np.argsort(p_active)[::-1]]

    print(f"\n── Top {args.top_n} Most Pure Clusters ──")
    print(f"{'Cluster':>8}  {'Top Species':>12}  {'P(top)':>7}  {'Purity':>7}  {'N':>6}")
    print("-" * 50)
    for ci in sorted_by_purity[:args.top_n]:
        top_sp_idx = probs[ci].argmax()
        print(f"  {ci:>6}  {species_names[top_sp_idx]:>12}  "
              f"{probs[ci, top_sp_idx]:>7.3f}  {purity[ci]:>7.3f}  "
              f"{total_per_cluster[ci]:>6.0f}")

    # ── 3. Most mixed / noisy clusters ───────────────────────────────────────
    print(f"\n── Bottom {args.top_n} Most Mixed Clusters ──")
    print(f"{'Cluster':>8}  {'Top Species':>12}  {'P(top)':>7}  {'Purity':>7}  {'N':>6}")
    print("-" * 50)
    for ci in sorted_by_purity[-args.top_n:]:
        top_sp_idx = probs[ci].argmax()
        print(f"  {ci:>6}  {species_names[top_sp_idx]:>12}  "
              f"{probs[ci, top_sp_idx]:>7.3f}  {purity[ci]:>7.3f}  "
              f"{total_per_cluster[ci]:>6.0f}")

    # ── 4. Per-species cluster coverage ──────────────────────────────────────
    # For each species: how many clusters does it dominate?
    dominant_species = probs.argmax(axis=1)  # (K,) — which species owns each cluster
    species_cluster_counts = np.bincount(dominant_species[active], minlength=S)
    species_total_counts = counts[active].sum(axis=0)  # total segments per species

    sp_df = pd.DataFrame({
        "species": species_names,
        "dominant_clusters": species_cluster_counts,
        "total_segments": species_total_counts,
    }).sort_values("dominant_clusters", ascending=False)

    print(f"\n── Species Cluster Ownership (top {args.top_species}) ──")
    print(f"{'Species':>12}  {'Dominant Clusters':>18}  {'Total Segments':>15}")
    print("-" * 52)
    for _, row in sp_df.head(args.top_species).iterrows():
        print(f"  {row['species']:>12}  {int(row['dominant_clusters']):>18}  "
              f"{int(row['total_segments']):>15}")

    species_no_cluster = (sp_df["dominant_clusters"] == 0).sum()
    print(f"\n  Species with 0 dominant clusters: {species_no_cluster}/{S}")

    # ── 5. GMM sub-cluster summary ───────────────────────────────────────────
    if gmm_path.exists():
        gmm = pd.read_csv(gmm_path)
        print(f"\n── Within-Species GMM Summary ({len(gmm)} species) ──")
        print(f"  Components distribution:")
        for n in sorted(gmm["n_components"].unique()):
            cnt = (gmm["n_components"] == n).sum()
            print(f"    {n} components: {cnt} species ({100*cnt/len(gmm):.1f}%)")

        # Species with only 1 component (not enough samples for sub-clustering)
        low = gmm[gmm["n_components"] <= 1].sort_values("n_samples")
        print(f"\n  Species with ≤1 GMM component (low data): {len(low)}")
        if len(low) > 0:
            print(f"  {'Species':>12}  {'Samples':>8}")
            for _, r in low.head(10).iterrows():
                print(f"    {r['species']:>12}  {int(r['n_samples']):>8}")

        # Species with many sub-clusters (high acoustic diversity)
        rich = gmm[gmm["n_components"] >= 5].sort_values("n_components", ascending=False)
        if len(rich) > 0:
            print(f"\n  Species with ≥5 GMM components (acoustically diverse): {len(rich)}")
            print(f"  {'Species':>12}  {'Components':>11}  {'Samples':>8}")
            for _, r in rich.head(10).iterrows():
                print(f"    {r['species']:>12}  {int(r['n_components']):>11}  "
                      f"{int(r['n_samples']):>8}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
