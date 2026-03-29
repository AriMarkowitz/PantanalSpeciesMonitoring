"""Stage 2: Two-level motif discovery.

Level 1 — Global HDBSCAN: discover recurring acoustic motifs across
all spatial embeddings (label-free).

Level 2 — Within-species GMM: discover distinct call types per species
using BIC-selected Gaussian Mixture Models.

Usage:
    python src/cluster.py
    python src/cluster.py --set stage2.global.subsample_n=100000

    # Include distill embeddings in both clustering levels:
    python src/cluster.py --with-distill
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from tqdm import tqdm

from config import get_config
from utils import setup_logging, build_label_map, parse_soundscape_labels


# ─────────────────────────────────────────────
# Helpers for optional distill merge
# ─────────────────────────────────────────────

def _open_h5_pair(primary_h5: str, distill_h5: str | None, logger):
    """Open primary HDF5 and optionally a distill HDF5.

    Returns (h5_primary, h5_distill_or_None, n_primary, n_distill).
    The caller is responsible for closing both files.
    """
    h5p = h5py.File(primary_h5, "r")
    written_p = h5p["written"][:]
    n_primary = int(written_p.sum())

    h5d = None
    n_distill = 0
    if distill_h5 and Path(distill_h5).exists():
        h5d = h5py.File(distill_h5, "r")
        written_d = h5d["written"][:]
        n_distill = int(written_d.sum())
        logger.info(f"Merging distill embeddings: {n_distill} valid segments "
                    f"from {distill_h5}")
    elif distill_h5:
        logger.warning(f"--with-distill specified but {distill_h5} not found; "
                       f"proceeding with primary embeddings only")

    logger.info(f"Primary valid embeddings: {n_primary}"
                + (f", distill: {n_distill}" if n_distill else ""))
    return h5p, h5d, written_p, n_primary, n_distill


def _load_spatial_sample(h5f, written, subsample_n, rng, logger):
    """Return a (M, 1536) array sampled from spatial or global embeddings."""
    valid_indices = np.where(written)[0]
    has_spatial = "spatial_embeddings" in h5f
    n_per_segment = 15 if has_spatial else 1

    n_needed = subsample_n // n_per_segment + 1
    if n_needed >= len(valid_indices):
        sampled = valid_indices
    else:
        sampled = rng.choice(valid_indices, n_needed, replace=False)

    chunks = []
    if has_spatial:
        for i in range(0, len(sampled), 500):
            batch = sorted(sampled[i:i + 500])
            for idx in batch:
                chunks.append(h5f["spatial_embeddings"][idx].reshape(-1, 1536))
    else:
        chunks.append(h5f["global_embeddings"][sorted(sampled)])

    return np.vstack(chunks)


# ─────────────────────────────────────────────
# Level 1: Global Motif Discovery
# ─────────────────────────────────────────────

def cluster_global(cfg: dict, logger, distill_h5: str | None = None):
    """Discover global motif prototypes via HDBSCAN on spatial embeddings."""
    import umap
    import hdbscan

    gcfg = cfg["stage2"]["global"]
    h5_path = cfg["outputs"]["embeddings_h5"]
    out_dir = Path(cfg["outputs"]["prototypes_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    subsample_n = gcfg["subsample_n"]

    h5p, h5d, written_p, n_primary, n_distill = _open_h5_pair(
        h5_path, distill_h5, logger
    )
    try:
        # Allocate subsample budget proportionally if merging
        if h5d is not None and n_distill > 0:
            total_valid = n_primary + n_distill
            budget_p = int(subsample_n * n_primary / total_valid)
            budget_d = subsample_n - budget_p
        else:
            budget_p = subsample_n
            budget_d = 0

        logger.info(f"Loading primary spatial sample (budget={budget_p})")
        local_p = _load_spatial_sample(h5p, written_p, budget_p, rng, logger)

        if h5d is not None and budget_d > 0:
            written_d = h5d["written"][:]
            logger.info(f"Loading distill spatial sample (budget={budget_d})")
            local_d = _load_spatial_sample(h5d, written_d, budget_d, rng, logger)
            all_local = np.vstack([local_p, local_d])
        else:
            all_local = local_p

        if len(all_local) > subsample_n:
            all_local = all_local[rng.choice(len(all_local), subsample_n,
                                             replace=False)]
    finally:
        h5p.close()
        if h5d is not None:
            h5d.close()

    logger.info(f"Subsample shape: {all_local.shape}")

    # UMAP dimensionality reduction
    umap_dim = gcfg["umap_dim"]
    emb_dim = all_local.shape[1]
    logger.info(f"UMAP reduction: {emb_dim}D → {umap_dim}D")
    reducer = umap.UMAP(n_components=umap_dim, n_neighbors=30, min_dist=0.0,
                        metric="cosine", random_state=42, verbose=True)
    reduced = reducer.fit_transform(all_local)
    logger.info(f"UMAP complete: {reduced.shape}")

    # HDBSCAN clustering
    min_cluster_size = gcfg["min_cluster_size"]
    min_samples = gcfg["min_samples"]
    logger.info(f"HDBSCAN: min_cluster_size={min_cluster_size}, "
                f"min_samples={min_samples}")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    logger.info(f"Found {n_clusters} clusters, {n_noise} noise points "
                f"({100*n_noise/len(labels):.1f}%)")

    # Compute medoids in original 1536-D space (not UMAP space)
    logger.info("Computing cluster medoids in original embedding space")
    prototypes = []
    prototype_labels = []
    for k in range(n_clusters):
        mask = labels == k
        cluster_points = all_local[mask]
        # Medoid: point closest to centroid
        centroid = cluster_points.mean(axis=0)
        dists = np.linalg.norm(cluster_points - centroid, axis=1)
        medoid = cluster_points[dists.argmin()]
        prototypes.append(medoid)
        prototype_labels.append(k)

    prototypes = np.array(prototypes)  # (K, 1536)
    logger.info(f"Prototype matrix: {prototypes.shape}")

    # Save artifacts
    np.savez(out_dir / "global_prototypes.npz",
             prototypes=prototypes,
             labels=np.array(prototype_labels))

    with open(out_dir / "global_hdbscan.pkl", "wb") as f:
        pickle.dump(clusterer, f)

    with open(out_dir / "global_umap.pkl", "wb") as f:
        pickle.dump(reducer, f)

    # Save cluster stats
    cluster_sizes = np.bincount(labels[labels >= 0])
    stats = pd.DataFrame({
        "cluster_id": range(n_clusters),
        "size": cluster_sizes,
    })
    stats.to_csv(out_dir / "global_cluster_stats.csv", index=False)

    logger.info(f"Global clustering complete. {n_clusters} prototypes saved "
                f"to {out_dir}")
    return prototypes, clusterer, reducer


# ─────────────────────────────────────────────
# Level 2: Within-Species Sub-Clustering
# ─────────────────────────────────────────────

def cluster_within_species(cfg: dict, logger, distill_h5: str | None = None):
    """Fit per-species GMMs with BIC-based model selection.

    If distill_h5 is provided, labeled distill segments are merged with
    primary segments to enrich the per-species embedding pools.
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import PCA

    scfg = cfg["stage2"]["species"]
    h5_path = cfg["outputs"]["embeddings_h5"]
    segments_csv = cfg["outputs"]["segments_csv"]
    out_dir = Path(cfg["outputs"]["prototypes_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load primary segments metadata
    segments = pd.read_csv(segments_csv, low_memory=False)
    labeled = segments[segments["is_labeled"] == True].copy()

    # Build species -> segment indices mapping
    # For XC/iNat: primary_label is a single species
    # For soundscapes: primary_label is semicolon-separated
    species_to_indices = {}  # {species: [primary_seg_indices]}
    for idx, row in labeled.iterrows():
        labels = str(row["primary_label"]).split(";")
        for sp in labels:
            sp = sp.strip()
            if sp and sp != "nan":
                species_to_indices.setdefault(sp, []).append(idx)

    logger.info(f"Found {len(species_to_indices)} species with labeled data "
                f"(primary)")

    # Optionally load distill segments and build a parallel index
    distill_species_to_indices: dict[str, list[int]] = {}
    h5d = None
    written_d: np.ndarray | None = None
    if distill_h5 and Path(distill_h5).exists():
        distill_seg_csv = cfg["outputs"].get(
            "distill_segments_csv", "outputs/distill_segments.csv"
        )
        if Path(distill_seg_csv).exists():
            dseg = pd.read_csv(distill_seg_csv, low_memory=False)
            dlabeled = dseg[dseg["is_labeled"] == True].copy()
            for idx, row in dlabeled.iterrows():
                sp = str(row["primary_label"]).strip()
                if sp and sp != "nan":
                    distill_species_to_indices.setdefault(sp, []).append(idx)
            h5d = h5py.File(distill_h5, "r")
            written_d = h5d["written"][:]
            n_added = sum(
                1 for sp in distill_species_to_indices
                if sp in species_to_indices
            )
            logger.info(
                f"Distill labeled segments: "
                f"{len(distill_species_to_indices)} species, "
                f"{n_added} overlap with primary taxonomy"
            )
        else:
            logger.warning(
                f"distill_h5 specified but {distill_seg_csv} not found; "
                f"skipping distill merge for GMM"
            )

    # Load primary global embeddings
    h5f = h5py.File(h5_path, "r")
    global_emb = h5f["global_embeddings"]
    written = h5f["written"][:]

    pca_dim = scfg["pca_dim"]
    max_K = scfg["max_components"]
    min_per_comp = scfg["min_samples_per_component"]

    # Merge species sets (primary + distill)
    all_species = set(species_to_indices) | set(distill_species_to_indices)
    species_models = {}  # {species_id: {"pca": PCA, "gmm": GMM, "n_components": K}}

    for sp in tqdm(all_species, desc="Within-species GMM"):
        # Primary valid embeddings
        p_idx = species_to_indices.get(sp, [])
        valid_p = [i for i in p_idx if i < len(written) and written[i]]
        emb_p = global_emb[sorted(valid_p)] if valid_p else np.empty((0, 1536), dtype=np.float32)

        # Distill valid embeddings
        if h5d is not None and written_d is not None:
            d_idx = distill_species_to_indices.get(sp, [])
            valid_d = [i for i in d_idx if i < len(written_d) and written_d[i]]
            emb_d = (h5d["global_embeddings"][sorted(valid_d)]
                     if valid_d else np.empty((0, 1536), dtype=np.float32))
        else:
            valid_d = []
            emb_d = np.empty((0, 1536), dtype=np.float32)

        # Concatenate
        emb_parts = [e for e in [emb_p, emb_d] if len(e) > 0]
        if not emb_parts:
            species_models[sp] = {"n_components": 0, "n_samples": 0}
            continue
        emb = np.vstack(emb_parts) if len(emb_parts) > 1 else emb_parts[0]
        valid = valid_p + valid_d

        if len(valid) < 2:
            species_models[sp] = {"n_components": 0, "n_samples": len(valid)}
            continue

        # PCA
        n_comp_pca = min(pca_dim, emb.shape[0] - 1, emb.shape[1])
        if n_comp_pca < 2:
            species_models[sp] = {"n_components": 0, "n_samples": len(valid)}
            continue

        pca = PCA(n_components=n_comp_pca)
        reduced = pca.fit_transform(emb)

        # BIC model selection
        K_max = min(max_K, len(valid) // min_per_comp)
        K_max = max(K_max, 1)

        best_gmm = None
        best_bic = float("inf")
        best_k = 1

        for k in range(1, K_max + 1):
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    random_state=42,
                    max_iter=200,
                )
                gmm.fit(reduced)
                bic = gmm.bic(reduced)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
                    best_k = k
            except Exception:
                break  # Usually means k is too large for the data

        species_models[sp] = {
            "pca": pca,
            "gmm": best_gmm,
            "n_components": best_k,
            "n_samples": len(valid),
        }

    h5f.close()
    if h5d is not None:
        h5d.close()

    # Summary
    component_counts = [m["n_components"] for m in species_models.values()]
    logger.info(f"Within-species clustering complete:")
    logger.info(f"  Species processed: {len(species_models)}")
    logger.info(f"  Components distribution: "
                f"min={min(component_counts)}, "
                f"max={max(component_counts)}, "
                f"mean={np.mean(component_counts):.1f}")

    # Save
    with open(out_dir / "species_gmms.pkl", "wb") as f:
        pickle.dump(species_models, f)

    # Save summary CSV
    summary = pd.DataFrame([
        {"species": sp, "n_samples": m["n_samples"],
         "n_components": m["n_components"]}
        for sp, m in species_models.items()
    ])
    summary.to_csv(out_dir / "species_gmm_summary.csv", index=False)

    logger.info(f"Saved species GMMs to {out_dir / 'species_gmms.pkl'}")
    return species_models


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--with-distill", action="store_true",
                   help="Merge distill embeddings into clustering")
    p.add_argument("--distill-h5", type=str, default=None,
                   help="Path to distill_embeddings.h5 "
                        "(default: outputs/embeddings/distill_embeddings.h5)")
    known, _ = p.parse_known_args()
    return known


def main(cfg: dict):
    args = parse_args()
    logger = setup_logging("cluster", cfg["outputs"]["logs_dir"])
    logger.info("Stage 2: Motif Discovery")

    distill_h5 = None
    if args.with_distill:
        distill_h5 = args.distill_h5 or cfg["outputs"].get(
            "distill_embeddings_h5", "outputs/embeddings/distill_embeddings.h5"
        )
        logger.info(f"Distill embeddings: {distill_h5}")

    logger.info("=== Level 1: Global Motif Discovery (HDBSCAN) ===")
    prototypes, clusterer, reducer = cluster_global(cfg, logger,
                                                    distill_h5=distill_h5)

    logger.info("=== Level 2: Within-Species Sub-Clustering (GMM) ===")
    species_models = cluster_within_species(cfg, logger,
                                            distill_h5=distill_h5)

    logger.info("Stage 2 complete")


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
