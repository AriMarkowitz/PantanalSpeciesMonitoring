"""Stage 3: Motif assignment and feature vector construction.

Reads embeddings (Stage 1) and prototypes/GMMs (Stage 2), computes per-segment
feature vectors combining global embeddings, motif features, sub-cluster
likelihoods, and Perch logits. Writes a single features.h5 with labels,
masks, and fold assignments ready for training.

Usage:
    python src/build_features.py
"""

import pickle
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from scipy.special import softmax
from scipy.stats import entropy
import sys
from tqdm import tqdm

_IN_TTY = sys.stdout.isatty()

from config import get_config
from cluster import load_supcon_projection
from utils import setup_logging, build_label_map, parse_soundscape_labels


def load_prototypes(proto_dir: str):
    """Load global prototypes and HDBSCAN artifacts."""
    proto_dir = Path(proto_dir)
    data = np.load(proto_dir / "global_prototypes.npz")
    prototypes = data["prototypes"]  # (K, 1536)

    hdbscan_path = proto_dir / "global_hdbscan.pkl"
    clusterer = None
    if hdbscan_path.exists():
        with open(hdbscan_path, "rb") as f:
            clusterer = pickle.load(f)

    return prototypes, clusterer


def load_species_gmms(proto_dir: str):
    """Load per-species GMM models."""
    path = Path(proto_dir) / "species_gmms.pkl"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8):
    """L2-normalize along an axis."""
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norms, eps)


def compute_global_motif_features(spatial_emb: np.ndarray,
                                  prototypes: np.ndarray,
                                  temperature: float = 1.0,
                                  metric: str = "cosine"):
    """Compute soft motif assignment features for one segment.

    Args:
        spatial_emb: (5, 3, 1536) or (15, 1536) spatial embeddings
        prototypes: (K, 1536) global prototypes
        temperature: softmax temperature
        metric: "cosine" (default) or "euclidean"

    Returns:
        motif_histogram: (K,)
        max_activation: (K,)
        temporal_spread: (K,)
        noise_fraction: scalar
    """
    local = spatial_emb.reshape(-1, spatial_emb.shape[-1])  # (15, 1536)
    K = prototypes.shape[0]

    if metric == "cosine":
        # Cosine similarity: natural for contrastive embeddings like Perch
        local_n = _l2_normalize(local, axis=1)      # (15, 1536)
        proto_n = _l2_normalize(prototypes, axis=1)  # (K, 1536)
        sim = local_n @ proto_n.T                     # (15, K) in [-1, 1]
        logits = sim / temperature
    else:
        # Euclidean (squared) distance — negate so closer = higher
        z_sq = np.sum(local ** 2, axis=1, keepdims=True)
        p_sq = np.sum(prototypes ** 2, axis=1, keepdims=True).T
        dists = z_sq + p_sq - 2.0 * local @ prototypes.T
        logits = -dists / temperature

    weights = softmax(logits, axis=1)  # (15, K)

    motif_histogram = weights.sum(axis=0)  # (K,)
    max_activation = weights.max(axis=0)   # (K,)
    temporal_spread = weights.std(axis=0)  # (K,)

    # Noise fraction: high entropy across all clusters = likely noise
    avg_entropy = np.mean([entropy(w) for w in weights])
    max_possible_entropy = np.log(K) if K > 1 else 1.0
    noise_fraction = avg_entropy / max_possible_entropy

    return motif_histogram, max_activation, temporal_spread, noise_fraction


def compute_global_motif_features_batched(
    spatial_embs: np.ndarray,
    prototypes: np.ndarray,
    temperature: float = 0.1,
    metric: str = "cosine",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized version of compute_global_motif_features over N segments.

    Args:
        spatial_embs: (N, 15, 1536) or (N, 1536) spatial embeddings
        prototypes: (K, 1536) global prototypes

    Returns:
        hist:    (N, K)
        max_act: (N, K)
        spread:  (N, K)
        noise:   (N,)
    """
    if spatial_embs.ndim == 2:
        # global-only fallback: treat each segment as (1, 1536)
        spatial_embs = spatial_embs[:, np.newaxis, :]  # (N, 1, 1536)

    N, L, D = spatial_embs.shape  # L = number of local patches (15)
    K = prototypes.shape[0]

    # Flatten to (N*L, D), compute similarities, reshape back
    local = spatial_embs.reshape(N * L, D)

    if metric == "cosine":
        local_n = local / (np.linalg.norm(local, axis=1, keepdims=True) + 1e-8)
        proto_n = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
        logits = (local_n @ proto_n.T) / temperature          # (N*L, K)
    else:
        z_sq = np.sum(local ** 2, axis=1, keepdims=True)
        p_sq = np.sum(prototypes ** 2, axis=1, keepdims=True).T
        dists = z_sq + p_sq - 2.0 * local @ prototypes.T
        logits = -dists / temperature                          # (N*L, K)

    weights = softmax(logits, axis=1).reshape(N, L, K)         # (N, L, K)

    hist    = weights.sum(axis=1)                              # (N, K)
    max_act = weights.max(axis=1)                              # (N, K)
    spread  = weights.std(axis=1)                              # (N, K)

    # Noise fraction per segment
    avg_ent = -np.sum(weights * np.log(weights + 1e-10), axis=2).mean(axis=1)  # (N,)
    max_ent = np.log(K) if K > 1 else 1.0
    noise = avg_ent / max_ent                                  # (N,)

    return hist, max_act, spread, noise


def compute_species_subcluster_features(global_emb: np.ndarray,
                                        species_gmms: dict,
                                        label_map: dict):
    """Compute per-species sub-cluster match features for a single segment.

    Args:
        global_emb: (1536,) global embedding for one segment
        species_gmms: {species_id: {"pca": PCA, "gmm": GMM, ...}}
        label_map: {species_id: index}

    Returns:
        best_match: (num_species,) best sub-cluster posterior per species
        sub_entropy: (num_species,) entropy of sub-cluster posteriors per species
    """
    num_species = len(label_map)
    best_match = np.full(num_species, -100.0, dtype=np.float32)
    sub_entropy = np.zeros(num_species, dtype=np.float32)

    emb_2d = global_emb.reshape(1, -1)

    for sp, idx in label_map.items():
        if sp not in species_gmms:
            continue
        model = species_gmms[sp]
        if model["n_components"] == 0 or "gmm" not in model or model["gmm"] is None:
            continue

        try:
            pca = model["pca"]
            gmm = model["gmm"]
            reduced = pca.transform(emb_2d)
            posteriors = gmm.predict_proba(reduced)[0]  # (K_s,)
            best_match[idx] = posteriors.max()
            sub_entropy[idx] = entropy(posteriors + 1e-10)
        except Exception:
            continue

    return best_match, sub_entropy


def compute_all_subcluster_features_batched(
    global_embs: np.ndarray,
    species_gmms: dict,
    label_map: dict,
    logger=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized per-species GMM scoring over all N segments at once.

    Instead of calling pca.transform+gmm.predict_proba N×234 times (83M
    sklearn calls at ~28h), this loops over 234 species and for each one
    batch-processes all N embeddings in a single sklearn call.

    Args:
        global_embs: (N, 1536) all global embeddings (valid segments only;
                     caller stores valid_indices for scatter-back)
        species_gmms: {species_id: {"pca": PCA, "gmm": GMM, ...}}
        label_map: {species_id: index}

    Returns:
        best_match: (N, num_species) float32
        sub_entropy: (N, num_species) float32
    """
    N = global_embs.shape[0]
    num_species = len(label_map)
    best_match = np.full((N, num_species), -100.0, dtype=np.float32)
    sub_entropy = np.zeros((N, num_species), dtype=np.float32)

    for sp, idx in tqdm(label_map.items(), desc="GMM species", total=num_species,
                        leave=False):
        if sp not in species_gmms:
            continue
        model = species_gmms[sp]
        if model["n_components"] == 0 or "gmm" not in model or model["gmm"] is None:
            continue

        try:
            pca = model["pca"]
            gmm = model["gmm"]
            reduced = pca.transform(global_embs)          # (N, pca_dim)
            posteriors = gmm.predict_proba(reduced)        # (N, K_s)
            best_match[:, idx] = posteriors.max(axis=1)
            sub_entropy[:, idx] = entropy(posteriors.T + 1e-10)  # entropy over K_s
        except Exception:
            continue

    return best_match, sub_entropy


def build_labels_and_masks(segments: pd.DataFrame, label_map: dict):
    """Build multi-hot label matrix and loss masks from segment metadata.

    Returns:
        labels: (N, num_species) float32 multi-hot
        masks: (N, num_species) float32 — 1.0 where loss should be computed
    """
    num_species = len(label_map)
    N = len(segments)
    labels = np.zeros((N, num_species), dtype=np.float32)
    masks = np.zeros((N, num_species), dtype=np.float32)

    for i, row in segments.iterrows():
        quality = row["label_quality"]

        if quality == "unlabeled":
            # No loss signal — mask everything.
            # These are soundscape segments without annotations; we can't
            # assume species absence (no label ≠ species not present).
            continue

        elif quality == "strong_multilabel":
            # Soundscape: all species are supervised (absence = true negative)
            masks[i, :] = 1.0
            for sp in str(row["primary_label"]).split(";"):
                sp = sp.strip()
                if sp in label_map:
                    labels[i, label_map[sp]] = 1.0

        elif quality == "strong_primary":
            primary = str(row["primary_label"]).strip()
            if primary in label_map:
                # Target species focal recording: primary is positive,
                # secondaries are positive, all other targets are supervised
                # as negatives (focal recordings are dominated by the
                # primary species — safe to assume other targets are absent).
                masks[i, :] = 1.0
                labels[i, label_map[primary]] = 1.0

                # Secondary labels: positive where listed
                secondaries = str(row.get("secondary_labels", ""))
                if secondaries and secondaries != "nan":
                    for sp in secondaries.split(";"):
                        sp = sp.strip()
                        if sp in label_map:
                            labels[i, label_map[sp]] = 1.0
            else:
                # Non-target species focal recording: primary is NOT in
                # the competition taxonomy. This recording is dominated by
                # a non-target sound, so all 234 target species are treated
                # as confirmed negatives — teaches the model to reject
                # confusable non-target species.
                masks[i, :] = 1.0
                # labels already all zeros → all targets negative

    return labels, masks


def assign_folds(segments: pd.DataFrame, n_folds: int, seed: int = 42):
    """Assign fold indices ensuring no soundscape file leaks across folds."""
    folds = np.full(len(segments), -1, dtype=np.int8)
    rng = np.random.RandomState(seed)

    # Group soundscape segments by source file
    sc_mask = segments["source_type"] == "soundscape"
    if sc_mask.any():
        sc_files = segments.loc[sc_mask, "source_file"].unique()
        rng.shuffle(sc_files)
        file_to_fold = {f: i % n_folds for i, f in enumerate(sc_files)}
        for idx in segments.index[sc_mask]:
            folds[idx] = file_to_fold[segments.loc[idx, "source_file"]]

    # Non-soundscape: stratified by primary label
    non_sc = segments.index[~sc_mask]
    if len(non_sc) > 0:
        shuffled = rng.permutation(non_sc)
        for i, idx in enumerate(shuffled):
            folds[idx] = i % n_folds

    return folds


def main(cfg: dict):
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--student", action="store_true",
                        help="Use student embeddings instead of Perch")
    args, _ = parser.parse_known_args()

    logger = setup_logging("build_features", cfg["outputs"]["logs_dir"])
    logger.info("Stage 3: Motif Assignment & Feature Extraction")

    # Load inputs
    segments = pd.read_csv(cfg["outputs"]["segments_csv"], low_memory=False)
    label_map = build_label_map(cfg["data"]["taxonomy_csv"])
    num_species = len(label_map)
    logger.info(f"Segments: {len(segments)}, Species: {num_species}")

    prototypes, clusterer = load_prototypes(cfg["outputs"]["prototypes_dir"])
    K_global = prototypes.shape[0]
    logger.info(f"Global prototypes: {K_global}")

    # Check if prototypes were built with SupCon projection
    proto_data = np.load(
        Path(cfg["outputs"]["prototypes_dir"]) / "global_prototypes.npz",
        allow_pickle=True,
    )
    used_supcon = bool(proto_data.get("used_supcon", False))
    project_fn = None
    if used_supcon:
        project_fn = load_supcon_projection(
            cfg["outputs"]["prototypes_dir"], logger
        )
        if project_fn is None:
            logger.error("Prototypes were built with SupCon but projection "
                         "matrix not found!")
            return
        logger.info("SupCon projection will be applied to embeddings "
                    "for motif feature computation")

    species_gmms = load_species_gmms(cfg["outputs"]["prototypes_dir"])
    logger.info(f"Species GMMs: {len(species_gmms)}")

    # Choose embeddings source: student or Perch
    if args.student:
        emb_path = str(Path(cfg["outputs"]["embeddings_h5"]).parent / "student_embeddings.h5")
        logger.info(f"Using STUDENT embeddings: {emb_path}")
    else:
        emb_path = cfg["outputs"]["embeddings_h5"]
        logger.info(f"Using Perch embeddings: {emb_path}")

    h5_emb = h5py.File(emb_path, "r")
    written = h5_emb["written"][:]
    has_spatial = "spatial_embeddings" in h5_emb

    # Optional: per-class NMF features (stage_nmf_pc)
    # Adds 2 * num_species features per segment: [recon_error_sp, activation_energy_sp]
    nmf_features = None
    nmf_dir = Path(cfg["outputs"].get("nmf_dir", ""))
    nmf_path = nmf_dir / "nmf_features.npy"
    if nmf_dir and nmf_path.exists():
        nmf_features = np.load(nmf_path)  # (N, 2*num_species)
        if nmf_features.shape[0] != len(segments):
            logger.warning(
                f"NMF features shape mismatch: {nmf_features.shape[0]} rows "
                f"vs {len(segments)} segments. Skipping NMF features."
            )
            nmf_features = None
        else:
            logger.info(f"Loaded NMF features: {nmf_features.shape} from {nmf_path}")

    # Feature dimensionality
    # NOTE: Perch logits were previously included here, but Perch cannot run
    # on Kaggle CPU at inference time, so they are dropped entirely from the
    # feature vector (rather than zero-filling and wasting 200 MLP inputs).
    feat_dim = (
        1536                    # global embedding
        + 3 * K_global          # motif histogram + max + spread
        + 1                     # noise fraction
        + 2 * num_species       # best subcluster match + entropy
    )
    if nmf_features is not None:
        feat_dim += nmf_features.shape[1]   # 2 * num_species (recon + energy)
    logger.info(f"Feature dimension: {feat_dim}")

    # Build labels and masks
    logger.info("Building labels and loss masks")
    labels, masks = build_labels_and_masks(segments, label_map)

    # Assign folds
    n_folds = cfg["stage4"]["n_folds"]
    folds = assign_folds(segments, n_folds)

    # Create output HDF5
    out_path = cfg["outputs"]["features_h5"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    N = len(segments)

    # ── Batched GMM scoring (vectorized over all valid segments) ──────────────
    # Load all valid global embeddings into RAM once, then score all species
    # in 234 batch sklearn calls instead of N×234 per-segment calls (~200x faster).
    valid_indices = np.where(written)[0]
    logger.info(f"Loading {len(valid_indices)} valid global embeddings for batched GMM scoring")
    global_embs_valid = h5_emb["global_embeddings"][sorted(valid_indices)]  # (M, 1536)

    # GMMs were fitted on SupCon-projected embeddings (256D), so we must
    # project here too — otherwise PCA.transform fails on 1536D input.
    if project_fn is not None:
        gmm_embs = project_fn(global_embs_valid)  # (M, proj_dim)
        logger.info(f"Projected embeddings for GMM scoring: {global_embs_valid.shape} → {gmm_embs.shape}")
    else:
        gmm_embs = global_embs_valid

    logger.info("Running batched species GMM scoring (vectorized)")
    gmm_best_match_valid, gmm_sub_ent_valid = compute_all_subcluster_features_batched(
        gmm_embs, species_gmms, label_map, logger=logger
    )  # (M, num_species) each

    # Build scatter-back lookup: original index → position in valid array
    valid_pos = {int(idx): pos for pos, idx in enumerate(sorted(valid_indices))}

    logger.info(f"Creating features HDF5: {out_path}")
    h5_out = h5py.File(out_path, "w")
    feat_ds = h5_out.create_dataset(
        "features", shape=(N, feat_dim), dtype="float32",
        chunks=(min(1000, N), feat_dim), compression="lzf",
    )
    h5_out.create_dataset("labels", data=labels, compression="lzf")
    h5_out.create_dataset("masks", data=masks, compression="lzf")
    h5_out.create_dataset("folds", data=folds)
    dt = h5py.string_dtype()
    seg_ids = h5_out.create_dataset("segment_ids", shape=(N,), dtype=dt)

    # Store feature layout as attributes
    h5_out.attrs["K_global"] = K_global
    h5_out.attrs["num_species"] = num_species
    h5_out.attrs["feat_dim"] = feat_dim

    # Process all segments — GMM features already computed, just look up
    temperature = cfg["stage3"].get("temperature", 0.1)
    metric = cfg["stage3"].get("similarity_metric", "cosine")

    # ── Segment IDs (vectorized, no .iloc loop) ──────────────────────────────
    seg_id_vals = segments["segment_id"].values
    for i, sid in enumerate(seg_id_vals):
        seg_ids[i] = sid

    # ── Bulk-load all valid embeddings ────────────────────────────────────────
    valid_idx_sorted = sorted(valid_indices)
    logger.info(f"Loading spatial embeddings for {len(valid_idx_sorted)} valid segments")

    global_embs_all = h5_emb["global_embeddings"][valid_idx_sorted]  # (M, 1536)

    if has_spatial:
        spatial_embs_all = h5_emb["spatial_embeddings"][valid_idx_sorted]  # (M, N_SPATIAL_FRAMES, 1536)
        sp = spatial_embs_all  # already (M, L, 1536)
    else:
        sp = global_embs_all[:, np.newaxis, :]  # (M, 1, 1536)

    # ── Batched motif features ────────────────────────────────────────────────
    logger.info("Computing motif features (batched)")
    CHUNK = 4096  # process in chunks to bound peak memory
    M = len(valid_idx_sorted)
    hist_all    = np.empty((M, K_global), dtype=np.float32)
    max_act_all = np.empty((M, K_global), dtype=np.float32)
    spread_all  = np.empty((M, K_global), dtype=np.float32)
    noise_all   = np.empty(M, dtype=np.float32)

    for start in tqdm(range(0, M, CHUNK), desc="Motif chunks", disable=not _IN_TTY):
        end = min(start + CHUNK, M)
        sp_chunk = sp[start:end]

        # If SupCon projection was used for clustering, project spatial
        # embeddings into the same space as the prototypes
        if project_fn is not None:
            orig_shape = sp_chunk.shape  # (chunk, L, 1536)
            sp_chunk = project_fn(sp_chunk)
            # Reshape back: project_fn flattens last dim, need (chunk, L, proj_dim)
            proj_dim = prototypes.shape[1]
            sp_chunk = sp_chunk.reshape(orig_shape[0], orig_shape[1], proj_dim)

        h, ma, s, n = compute_global_motif_features_batched(
            sp_chunk, prototypes, temperature=temperature, metric=metric,
        )
        hist_all[start:end]    = h
        max_act_all[start:end] = ma
        spread_all[start:end]  = s
        noise_all[start:end]   = n

    # ── Assemble and write feature matrix ────────────────────────────────────
    logger.info("Assembling and writing feature matrix")
    parts = [
        global_embs_all,       # (M, 1536)
        hist_all,              # (M, K_global)
        max_act_all,           # (M, K_global)
        spread_all,            # (M, K_global)
        noise_all[:, None],    # (M, 1)
        gmm_best_match_valid,  # (M, num_species)
        gmm_sub_ent_valid,     # (M, num_species)
    ]
    if nmf_features is not None:
        # nmf_features is (N, 2*num_species); slice to valid indices to align
        parts.append(nmf_features[valid_idx_sorted].astype(np.float32))
    feat_matrix = np.concatenate(parts, axis=1).astype(np.float32)

    # Write feature rows — use contiguous slice if all segments are valid (fast),
    # otherwise fall back to chunked fancy indexing (slower but correct).
    M = len(valid_idx_sorted)
    logger.info(f"Writing {M} feature rows to HDF5")
    all_contiguous = (M == N and valid_idx_sorted[0] == 0 and valid_idx_sorted[-1] == N - 1)
    if all_contiguous:
        logger.info("All segments valid — using contiguous slice write")
        WRITE_CHUNK = 50000
        for start in range(0, M, WRITE_CHUNK):
            end = min(start + WRITE_CHUNK, M)
            feat_ds[start:end] = feat_matrix[start:end]
            logger.info(f"  wrote {end}/{M} rows")
    else:
        logger.info("Sparse valid segments — using chunked fancy-index write")
        WRITE_CHUNK = 10000
        for start in range(0, M, WRITE_CHUNK):
            end = min(start + WRITE_CHUNK, M)
            feat_ds[valid_idx_sorted[start:end], :] = feat_matrix[start:end]
            logger.info(f"  wrote {end}/{M} rows")
    h5_out.flush()

    h5_emb.close()
    h5_out.flush()
    h5_out.close()

    logger.info(f"Features complete: {N} segments, {feat_dim}D → {out_path}")

    # Diagnostic: cluster→species association table
    # Reopens embeddings HDF5 read-only for a second pass over labeled data
    build_cluster_species_table(segments, labels, label_map, prototypes,
                                h5_emb_path=emb_path,
                                out_dir=Path(cfg["outputs"]["prototypes_dir"]),
                                logger=logger,
                                metric=cfg["stage3"].get("similarity_metric", "cosine"),
                                temperature=cfg["stage3"].get("temperature", 0.1),
                                project_fn=project_fn)

    logger.info("Stage 3 complete")


def build_cluster_species_table(segments: pd.DataFrame,
                                labels: np.ndarray,
                                label_map: dict,
                                prototypes: np.ndarray,
                                h5_emb_path: str,
                                out_dir: Path,
                                logger,
                                metric: str = "cosine",
                                temperature: float = 0.1,
                                project_fn=None):
    """Build P(species | cluster) table for diagnostics.

    For each global motif cluster, counts how often each species appears
    in segments that strongly activate that cluster. This is NOT used for
    classification (the MLP handles that nonlinearly), but for:
      - Sanity checking that clusters capture biologically meaningful structure
      - Debugging classifier failures
      - Giving ecologists an interpretable cluster→species view
    """
    logger.info("Building cluster→species association table")

    K = prototypes.shape[0]
    num_species = labels.shape[1]
    inv_label_map = {v: k for k, v in label_map.items()}

    h5f = h5py.File(h5_emb_path, "r")
    written = h5f["written"][:]
    has_spatial = "spatial_embeddings" in h5f

    # Accumulate: for each cluster, sum up the species label vectors
    # of segments where that cluster has the highest activation
    cluster_species_counts = np.zeros((K, num_species), dtype=np.float64)

    # Only use labeled segments
    labeled_mask = labels.sum(axis=1) > 0
    labeled_indices = np.where(labeled_mask & written)[0]

    logger.info(f"  Computing dominant cluster for {len(labeled_indices)} labeled segments")

    for i in tqdm(labeled_indices, desc="Cluster-species table", disable=not _IN_TTY):
        if has_spatial:
            spatial = h5f["spatial_embeddings"][i]  # (5, 3, 1536)
            local = spatial.reshape(-1, spatial.shape[-1])  # (15, 1536)
        else:
            local = h5f["global_embeddings"][i].reshape(1, -1)

        # Project into SupCon space if prototypes are in projected space
        if project_fn is not None:
            local = project_fn(local)

        # Compute soft assignment (same metric as feature extraction)
        if metric == "cosine":
            local_n = _l2_normalize(local, axis=1)
            proto_n = _l2_normalize(prototypes, axis=1)
            sim = local_n @ proto_n.T
            weights = softmax(sim / temperature, axis=1)
        else:
            dists = np.sum((local[:, None, :] - prototypes[None, :, :]) ** 2, axis=2)
            weights = softmax(-dists / temperature, axis=1)

        # Aggregate: motif histogram for this segment
        hist = weights.sum(axis=0)  # (K,)

        # Credit only the dominant cluster for this segment
        dominant = int(hist.argmax())
        cluster_species_counts[dominant] += labels[i]

    h5f.close()

    # Normalize to P(species | cluster)
    row_sums = cluster_species_counts.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    cluster_species_probs = cluster_species_counts / row_sums

    # Save full matrix
    np.savez(out_dir / "cluster_species_table.npz",
             counts=cluster_species_counts,
             probs=cluster_species_probs,
             species_names=np.array([inv_label_map.get(i, str(i))
                                     for i in range(num_species)]))

    # Log top species per cluster
    logger.info("  Top species per cluster:")
    for k in range(min(K, 20)):  # show first 20
        top_idx = np.argsort(cluster_species_probs[k])[::-1][:3]
        top_str = ", ".join(
            f"{inv_label_map.get(j, j)}={cluster_species_probs[k, j]:.2f}"
            for j in top_idx if cluster_species_probs[k, j] > 0.01
        )
        logger.info(f"    Cluster {k:3d} (n={cluster_species_counts[k].sum():.0f}): {top_str}")

    logger.info(f"  Saved to {out_dir / 'cluster_species_table.npz'}")


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
