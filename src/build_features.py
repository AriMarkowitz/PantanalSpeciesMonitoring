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
from tqdm import tqdm

from config import get_config
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


def compute_global_motif_features(spatial_emb: np.ndarray,
                                  prototypes: np.ndarray,
                                  temperature: float = 1.0):
    """Compute soft motif assignment features for one segment.

    Args:
        spatial_emb: (5, 3, 1536) or (15, 1536) spatial embeddings
        prototypes: (K, 1536) global prototypes
        temperature: softmax temperature

    Returns:
        motif_histogram: (K,)
        max_activation: (K,)
        temporal_spread: (K,)
        noise_fraction: scalar
    """
    local = spatial_emb.reshape(-1, spatial_emb.shape[-1])  # (15, 1536)
    K = prototypes.shape[0]

    # Compute pairwise squared distances: (15, K)
    # ||z - p||^2 = ||z||^2 + ||p||^2 - 2*z·p
    z_sq = np.sum(local ** 2, axis=1, keepdims=True)  # (15, 1)
    p_sq = np.sum(prototypes ** 2, axis=1, keepdims=True).T  # (1, K)
    dists = z_sq + p_sq - 2.0 * local @ prototypes.T  # (15, K)

    # Soft assignment via temperature-scaled softmax over clusters
    logits = -dists / temperature
    weights = softmax(logits, axis=1)  # (15, K)

    motif_histogram = weights.sum(axis=0)  # (K,)
    max_activation = weights.max(axis=0)   # (K,)
    temporal_spread = weights.std(axis=0)  # (K,)

    # Noise fraction: how spread out is the assignment?
    # High entropy across all clusters = likely noise
    avg_entropy = np.mean([entropy(w) for w in weights])
    max_possible_entropy = np.log(K) if K > 1 else 1.0
    noise_fraction = avg_entropy / max_possible_entropy

    return motif_histogram, max_activation, temporal_spread, noise_fraction


def compute_species_subcluster_features(global_emb: np.ndarray,
                                        species_gmms: dict,
                                        label_map: dict):
    """Compute per-species sub-cluster match features.

    Args:
        global_emb: (1536,) global embedding for one segment
        species_gmms: {species_id: {"pca": PCA, "gmm": GMM, ...}}
        label_map: {species_id: index}

    Returns:
        best_match: (num_species,) best sub-cluster log-likelihood per species
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
            # Best match: max posterior
            best_match[idx] = posteriors.max()
            # Entropy of sub-cluster assignment
            sub_entropy[idx] = entropy(posteriors + 1e-10)
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
            # No loss signal — mask everything
            continue

        elif quality == "strong_multilabel":
            # Soundscape: all species are supervised (absence = true negative)
            masks[i, :] = 1.0
            for sp in str(row["primary_label"]).split(";"):
                sp = sp.strip()
                if sp in label_map:
                    labels[i, label_map[sp]] = 1.0

        elif quality == "strong_primary":
            # XC/iNat: primary label is positive, secondaries are positive
            # but unlisted species are UNKNOWN (not negative)
            primary = str(row["primary_label"]).strip()
            if primary in label_map:
                idx = label_map[primary]
                labels[i, idx] = 1.0
                masks[i, idx] = 1.0  # only supervise the primary species

            # Secondary labels: positive where listed
            secondaries = str(row.get("secondary_labels", ""))
            if secondaries and secondaries != "nan":
                for sp in secondaries.split(";"):
                    sp = sp.strip()
                    if sp in label_map:
                        labels[i, label_map[sp]] = 1.0
                        masks[i, label_map[sp]] = 1.0

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

    species_gmms = load_species_gmms(cfg["outputs"]["prototypes_dir"])
    logger.info(f"Species GMMs: {len(species_gmms)}")

    h5_emb = h5py.File(cfg["outputs"]["embeddings_h5"], "r")
    written = h5_emb["written"][:]
    has_spatial = "spatial_embeddings" in h5_emb
    has_logits = "logit_values" in h5_emb
    top_k = cfg["stage3"]["top_k_logits"]

    # Feature dimensionality
    feat_dim = (
        1536                    # global embedding
        + 3 * K_global          # motif histogram + max + spread
        + 1                     # noise fraction
        + 2 * num_species       # best subcluster match + entropy
        + top_k                 # top Perch logits
    )
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
    h5_out.attrs["top_k_logits"] = top_k
    h5_out.attrs["feat_dim"] = feat_dim

    # Process all segments
    logger.info("Computing feature vectors")
    for i in tqdm(range(N), desc="Features"):
        seg_ids[i] = segments.iloc[i]["segment_id"]

        if not written[i]:
            # Embedding not available — leave as zeros
            continue

        # Global embedding
        global_emb = h5_emb["global_embeddings"][i]  # (1536,)

        # Motif features
        if has_spatial:
            spatial_emb = h5_emb["spatial_embeddings"][i]  # (5, 3, 1536)
        else:
            spatial_emb = global_emb.reshape(1, -1)  # fallback

        hist, max_act, spread, noise = compute_global_motif_features(
            spatial_emb, prototypes
        )

        # Sub-cluster features
        best_match, sub_ent = compute_species_subcluster_features(
            global_emb, species_gmms, label_map
        )

        # Perch logits
        if has_logits:
            logit_vals = h5_emb["logit_values"][i].astype(np.float32)  # (top_k,)
        else:
            logit_vals = np.zeros(top_k, dtype=np.float32)

        # Concatenate
        feat = np.concatenate([
            global_emb,        # 1536
            hist,              # K_global
            max_act,           # K_global
            spread,            # K_global
            [noise],           # 1
            best_match,        # num_species
            sub_ent,           # num_species
            logit_vals,        # top_k
        ])

        feat_ds[i] = feat

        # Flush periodically
        if i % 5000 == 0 and i > 0:
            h5_out.flush()

    h5_emb.close()
    h5_out.flush()
    h5_out.close()

    logger.info(f"Features complete: {N} segments, {feat_dim}D → {out_path}")


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
