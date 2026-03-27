"""Stage 2: Two-level motif discovery.

Level 1 — Global HDBSCAN: discover recurring acoustic motifs across
all spatial embeddings (label-free).

Level 2 — Within-species GMM: discover distinct call types per species
using BIC-selected Gaussian Mixture Models.

Usage:
    python src/cluster.py
    python src/cluster.py --set stage2.global.subsample_n=100000
"""

import pickle
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from tqdm import tqdm

from config import get_config
from utils import setup_logging, build_label_map, parse_soundscape_labels


# ─────────────────────────────────────────────
# Level 1: Global Motif Discovery
# ─────────────────────────────────────────────

def cluster_global(cfg: dict, logger):
    """Discover global motif prototypes via HDBSCAN on spatial embeddings."""
    import umap
    import hdbscan

    gcfg = cfg["stage2"]["global"]
    h5_path = cfg["outputs"]["embeddings_h5"]
    out_dir = Path(cfg["outputs"]["prototypes_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load spatial embeddings (memory-mapped)
    logger.info("Loading spatial embeddings from HDF5")
    h5f = h5py.File(h5_path, "r")
    written = h5f["written"][:]
    n_valid = int(written.sum())
    logger.info(f"Valid embeddings: {n_valid}")

    has_spatial = "spatial_embeddings" in h5f
    if has_spatial:
        spatial = h5f["spatial_embeddings"]  # (N, 5, 3, 1536) — keep lazy
        n_per_segment = 15  # 5 * 3
        emb_dim = 1536
    else:
        logger.warning("No spatial embeddings found, using global embeddings")
        spatial = None
        n_per_segment = 1
        emb_dim = 1536

    # Subsample for HDBSCAN
    subsample_n = min(gcfg["subsample_n"], n_valid * n_per_segment)
    logger.info(f"Subsampling {subsample_n} local embeddings for HDBSCAN")

    valid_indices = np.where(written)[0]
    rng = np.random.RandomState(42)

    if has_spatial:
        # Sample segment indices, then flatten spatial positions
        n_segments_needed = subsample_n // n_per_segment + 1
        if n_segments_needed >= len(valid_indices):
            sampled_seg_idx = valid_indices
        else:
            sampled_seg_idx = rng.choice(valid_indices, n_segments_needed,
                                         replace=False)

        # Load sampled spatial embeddings into memory
        logger.info(f"Loading {len(sampled_seg_idx)} segments' spatial embeddings")
        chunks = []
        for i in tqdm(range(0, len(sampled_seg_idx), 500), desc="Loading spatial"):
            batch_idx = sorted(sampled_seg_idx[i:i+500])
            for idx in batch_idx:
                chunk = spatial[idx]  # (5, 3, 1536)
                chunks.append(chunk.reshape(-1, emb_dim))  # (15, 1536)

        all_local = np.vstack(chunks)  # (M, 1536)
        if len(all_local) > subsample_n:
            all_local = all_local[rng.choice(len(all_local), subsample_n,
                                             replace=False)]
    else:
        # Use global embeddings directly
        sampled_seg_idx = rng.choice(valid_indices, min(subsample_n, len(valid_indices)),
                                     replace=False)
        all_local = h5f["global_embeddings"][sorted(sampled_seg_idx)]

    logger.info(f"Subsample shape: {all_local.shape}")

    # UMAP dimensionality reduction
    umap_dim = gcfg["umap_dim"]
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

    h5f.close()
    logger.info(f"Global clustering complete. {n_clusters} prototypes saved "
                f"to {out_dir}")
    return prototypes, clusterer, reducer


# ─────────────────────────────────────────────
# Level 2: Within-Species Sub-Clustering
# ─────────────────────────────────────────────

def cluster_within_species(cfg: dict, logger):
    """Fit per-species GMMs with BIC-based model selection."""
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import PCA

    scfg = cfg["stage2"]["species"]
    h5_path = cfg["outputs"]["embeddings_h5"]
    segments_csv = cfg["outputs"]["segments_csv"]
    out_dir = Path(cfg["outputs"]["prototypes_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load segments metadata
    segments = pd.read_csv(segments_csv, low_memory=False)
    labeled = segments[segments["is_labeled"] == True].copy()

    # Build species -> segment indices mapping
    # For XC/iNat: primary_label is a single species
    # For soundscapes: primary_label is semicolon-separated
    species_to_indices = {}
    for idx, row in labeled.iterrows():
        labels = str(row["primary_label"]).split(";")
        for sp in labels:
            sp = sp.strip()
            if sp and sp != "nan":
                if sp not in species_to_indices:
                    species_to_indices[sp] = []
                species_to_indices[sp].append(idx)

    logger.info(f"Found {len(species_to_indices)} species with labeled data")

    # Load global embeddings
    h5f = h5py.File(h5_path, "r")
    global_emb = h5f["global_embeddings"]
    written = h5f["written"][:]

    pca_dim = scfg["pca_dim"]
    max_K = scfg["max_components"]
    min_per_comp = scfg["min_samples_per_component"]

    species_models = {}  # {species_id: {"pca": PCA, "gmm": GMM, "n_components": K}}

    for sp, seg_indices in tqdm(species_to_indices.items(),
                                desc="Within-species GMM"):
        # Filter to valid (embedded) indices
        valid = [i for i in seg_indices if i < len(written) and written[i]]
        if len(valid) < 2:
            # Too few samples — single-component model
            species_models[sp] = {"n_components": 0, "n_samples": len(valid)}
            continue

        # Load embeddings for this species
        emb = global_emb[sorted(valid)]  # (n_s, 1536)

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

def main(cfg: dict):
    logger = setup_logging("cluster", cfg["outputs"]["logs_dir"])
    logger.info("Stage 2: Motif Discovery")

    logger.info("=== Level 1: Global Motif Discovery (HDBSCAN) ===")
    prototypes, clusterer, reducer = cluster_global(cfg, logger)

    logger.info("=== Level 2: Within-Species Sub-Clustering (GMM) ===")
    species_models = cluster_within_species(cfg, logger)

    logger.info("Stage 2 complete")


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
