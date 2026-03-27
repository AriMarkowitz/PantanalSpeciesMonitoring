# Pantanal Species Monitoring — Implementation Gameplan

**Date**: 2026-03-26
**Status**: DRAFT — awaiting review & approval

---

## Executive Summary

Build a multi-taxa bioacoustic classification system (birds, frogs, insects) for the Pantanal wetland using **Google Perch 2.0** as the backbone encoder. The system discovers recurring acoustic motifs via clustering, supports overlapping signals through soft assignment, and remains deployable on CPU via TFLite export.

This document maps the README architecture onto concrete implementation stages, integrates Perch 2.0's capabilities, provides Slurm job definitions for Easley HPC, and includes a critical evaluation with recommended changes.

---

## 1. Perch 2.0 Integration Plan

### 1.1 Model Specs

| Property | Value |
|----------|-------|
| Backbone | EfficientNet-B3 (~12M params) |
| Embedding dim | 1536 (global), spatial: `(5, 3, 1536)` |
| Input | 5s mono audio @ 32 kHz (160,000 samples) |
| Taxa coverage | ~15,000 species (birds, frogs, insects, mammals) |
| Frontend | PCEN mel-spectrogram (built-in) |
| Classification head | ~91M params, ~15K classes |
| Framework | TensorFlow 2.20+ (JAX-trained) |
| License | Apache 2.0 |

### 1.2 How Perch Replaces/Simplifies the README Architecture

The README proposes building a custom local encoder with multi-view spectrograms. **Perch 2.0 already does most of this better out of the box:**

- **Multi-view preprocessing (Section A)** → Perch uses PCEN internally, which handles background suppression. Custom multi-view channels are unnecessary for the encoder — Perch's training on millions of recordings already provides view invariance.
- **Local encoder (Section B)** → Perch's **spatial embeddings `(5, 3, 1536)`** are exactly the local embedding grid `Z = {z_1, ..., z_T}` described in the README. Each of the 15 spatial positions represents a local time-frequency receptive field.
- **Foundation model as teacher** → We skip this. Perch IS the foundation model. No distillation needed.

### 1.3 What Perch Does NOT Replace

- **Global clustering (Section C)** — still needed, applied to Perch spatial embeddings
- **Soft assignment (Section D)** — still needed, core novelty of the approach
- **Motif features (Section E)** — still needed
- **Classifier (Section F)** — still needed, but now much simpler (MLP on frozen embeddings + motif features)

### 1.4 GPU vs CPU Strategy

| Phase | Hardware | Rationale |
|-------|----------|-----------|
| Embedding extraction | **GPU** (L40S) | Batch-process all audio through Perch 2.0. One-time cost. |
| Clustering | **CPU** | HDBSCAN/k-means on extracted embeddings. Memory-bound, not GPU-bound. |
| Classifier training | **GPU** (L40S) | Train MLP head on embeddings + motif features. Fast. |
| Inference (deployment) | **CPU** | Perch 2.0 CPU release (Kaggle) + precomputed prototypes + lightweight classifier. |

**CPU inference**: Perch 2.0 now has an official CPU release on Kaggle, so TFLite conversion is no longer required for deployment. Install with `tensorflow-cpu` instead of `tensorflow[and-cuda]`. This also means we can run inference on Easley CPU partitions without GPU allocation.

---

## 2. Revised Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 0: Data Preparation                                        │
│   XC/iNat files + soundscapes → 5s segments → quality filtering  │
│   Output: segments + metadata CSV with label provenance          │
├──────────────────────────────────────────────────────────────────┤
│ STAGE 1: Embedding Extraction  [GPU]                             │
│   5s segments → Perch 2.0 → spatial embeddings (5,3,1536)        │
│                            → global embeddings (1536,)           │
│                            → logits (15K classes)                │
│   Output: HDF5 embedding database                                │
├──────────────────────────────────────────────────────────────────┤
│ STAGE 2: Motif Discovery  [CPU]                                  │
│   all spatial embeddings → HDBSCAN → prototypes P = {p_1..p_K}   │
│   Output: prototype vectors + cluster metadata                   │
├──────────────────────────────────────────────────────────────────┤
│ STAGE 3: Motif Assignment  [CPU]                                 │
│   per-file spatial embeddings + prototypes → soft assignments W   │
│   → motif histograms, temporal traces                            │
│   Output: motif feature vectors per file                         │
├──────────────────────────────────────────────────────────────────┤
│ STAGE 4: Classifier Training (initial)  [GPU]                    │
│   [global emb ∥ motif features ∥ Perch logits] → MLP             │
│   Train on: XC/iNat (masked loss) + labeled soundscapes          │
│   Output: initial classifier weights                             │
├──────────────────────────────────────────────────────────────────┤
│ STAGE 4b: Pseudo-Labeling Loop  [GPU]                            │
│   Round 1: Perch logits → high-conf pseudo-labels on unlabeled   │
│   Round 2+: classifier predictions → pseudo-labels → retrain     │
│   Repeat until val AUC stabilizes (2-3 rounds typical)           │
│   Output: final classifier weights + pseudo-label set            │
├──────────────────────────────────────────────────────────────────┤
│ STAGE 5: Inference / Deployment  [CPU]                           │
│   new audio → Perch CPU → embeddings → assign to prototypes      │
│   → motif features → classifier → species predictions            │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Implementation Stages (Detailed)

### Stage 0: Data Preparation

**Data sources** (two distinct types):

| Source | Format | Labels | Notes |
|--------|--------|--------|-------|
| **Xeno-Canto / iNaturalist** | Single-species focal recordings | Primary label (confirmed), secondary labels (may be incomplete) | Clean-ish, but secondary labels are unreliable — species may be present but unlabeled |
| **Soundscapes** | Long-duration field recordings | Partially labeled (some segments annotated, most not) | Realistic deployment conditions — overlapping species, noise, variable SNR |

**Label considerations**:
- XC/iNat primary labels are **strong positive** — the focal species is present
- XC/iNat secondary labels are **weak positive** — present but possibly incomplete (missing secondaries ≠ absent)
- Soundscape labeled segments are **strong multi-label** where annotated
- Unlabeled soundscape segments are **unknown** (not negative) — critical distinction for training

**Tasks**:
1. Inventory all audio files from both sources, validate formats (WAV/FLAC/MP3)
2. Resample to 32 kHz mono (Perch requirement)
3. Segment into 5-second non-overlapping windows (with optional 2.5s overlap for denser coverage)
4. Filter out silence/low-energy segments (simple energy threshold or Perch's own VAD via logits)
5. Build metadata CSV: `file_id, segment_id, start_time, end_time, path, source_type, primary_label, secondary_labels, is_labeled`
6. Track provenance: keep XC/iNat and soundscape data identifiable throughout the pipeline

**Output**: Directory of 5s WAV segments + metadata CSV with source and label provenance

### Stage 1: Embedding Extraction

**Tool**: `perch-hoplite` with `perch_v2` model

```python
from perch_hoplite.zoo import model_configs
import numpy as np, h5py, librosa

model = model_configs.load_model_by_name('perch_v2')

# Per segment:
waveform, _ = librosa.load(path, sr=32000, mono=True)
outputs = model.embed(waveform)
# Store: outputs.embeddings (1536,), spatial (5,3,1536), logits
```

**Storage**: HDF5 file with datasets:
- `global_embeddings`: `(N, 1536)` float32
- `spatial_embeddings`: `(N, 5, 3, 1536)` float32
- `logits`: `(N, ~15000)` float16 (compressed)
- `segment_ids`: string array

**Estimated size**: For 100K 5s segments → ~2.3 GB (spatial) + ~0.6 GB (global) + ~2.8 GB (logits float16) ≈ **~6 GB total**

### Stage 2: Motif Discovery (Two-Level Clustering)

Clustering operates at two levels with different goals and different algorithms.

#### Level 1: Global Motif Discovery (HDBSCAN)

**Goal**: Discover recurring acoustic patterns across the entire dataset — species calls, environmental textures, noise types — without any label information. Each discovered cluster is a "motif prototype."

**Input**: All spatial embeddings, flattened to `(N * 15, 1536)` — each 5s clip contributes 15 local embeddings from Perch's `(5, 3, 1536)` spatial grid.

**Approach**:

1. **Dimensionality reduction**: UMAP to 64D (speeds up clustering, denoises)
2. **Scalable HDBSCAN** — raw dataset is too large for direct HDBSCAN (O(n²) memory):
   ```
   Option A (recommended): Subsample → HDBSCAN → assign rest
     1. Random subsample ~200K embeddings (or stratified by file)
     2. HDBSCAN(min_cluster_size=50, min_samples=10, prediction_data=True)
     3. Use hdbscan.approximate_predict() to assign remaining points

   Option B: Two-tier clustering
     1. Mini-batch k-means with large K (e.g., 2000) on full dataset
     2. HDBSCAN on the 2000 centroids → meta-clusters
     3. Map back: each embedding inherits its k-means centroid's meta-cluster
   ```
3. **Soft membership**: Use `hdbscan.membership_vector(clusterer, points)` to get **probability vectors over all clusters** for each embedding — not hard labels. This is HDBSCAN's native soft clustering mode, returning `P(cluster_k | z_t)` for every point.

**What this produces**:
- `K_global` motif prototypes (cluster medoids in original 1536-D space, not UMAP space)
- Soft membership matrix: `(N * 15, K_global)` — each local embedding has a probability distribution over motifs
- Noise probability per embedding (HDBSCAN explicitly models "does not belong to any cluster")
- Many motifs will correlate strongly with specific species; some will be shared (e.g., "broadband noise," "water splash," "wind"); some will capture sub-species variation

**Key property**: A single 5s clip with a frog + bird will have its 15 spatial embeddings distributed across multiple motif clusters with varying soft memberships. The clip is **not forced into one cluster**.

#### Level 2: Within-Species Sub-Clustering (GMM)

**Goal**: For each labeled species, discover distinct vocalization types (song, alarm, contact call, juvenile, etc.). These become fine-grained templates — the classifier can learn that "species X, call type 3" looks like *this specific motif profile*.

**Input**: For each species `s`, gather all global embeddings (1536-D) from segments labeled with species `s`.

**Approach**:

1. **Per-species GMM with automatic K selection**:
   ```python
   from sklearn.mixture import GaussianMixture

   for species in species_list:
       embeddings_s = get_embeddings_for_species(species)  # (n_s, 1536)

       # Reduce dimensionality for GMM stability (1536-D is too high for full covariance)
       pca_s = PCA(n_components=min(64, n_s - 1)).fit_transform(embeddings_s)

       # BIC model selection: try K = 1..K_max, pick lowest BIC
       K_max = min(10, n_s // 20)  # at least 20 samples per component
       best_gmm = None
       best_bic = float('inf')
       for k in range(1, K_max + 1):
           gmm = GaussianMixture(n_components=k, covariance_type='full')
           gmm.fit(pca_s)
           if gmm.bic(pca_s) < best_bic:
               best_bic = gmm.bic(pca_s)
               best_gmm = gmm

       # Soft assignment: P(sub-cluster | embedding) for each sample
       posteriors = best_gmm.predict_proba(pca_s)  # (n_s, K_best)
   ```

2. **Why GMM here and not HDBSCAN**:
   - GMM natively returns soft posteriors — each embedding belongs to multiple sub-clusters with calibrated probabilities
   - BIC automatically selects K per species (no manual tuning)
   - Works with small N (species with 10 recordings get K=1, which is correct — one call type)
   - HDBSCAN would label most points as noise for rare species with <50 recordings

3. **What this produces per species**:
   - `K_s` sub-cluster components (means + covariances in PCA space)
   - Posterior matrix: `P(sub-cluster_j | embedding_i)` for each clip of that species
   - Interpretable: each sub-cluster ≈ a distinct vocalization type. Visualize by pulling top-5 exemplar spectrograms per sub-cluster.

**Example**: Species "Hypsiboas raniceps" (treefrog) might yield K=3:
- Sub-cluster 0: advertisement call (tonal, repetitive)
- Sub-cluster 1: aggressive call (broadband, short burst)
- Sub-cluster 2: rain chorus (overlapping with many conspecifics, lower SNR)

#### How the Two Levels Interact

The two levels are complementary, not redundant:

```
Global motifs (HDBSCAN)          Within-species sub-clusters (GMM)
────────────────────────         ────────────────────────────────
Label-free                       Label-dependent
Shared across species            Specific to one species
"What acoustic patterns exist?"  "What call types does species X have?"
Powers soft assignment features  Powers fine-grained matching
K_global ≈ 50-500               K_s ≈ 1-10 per species
```

At inference time, a new clip gets:
1. Global soft motif memberships → "this clip contains frog-like + insect-like patterns"
2. Per-species sub-cluster likelihoods → "if this is species X, it's most consistent with call type 2"

Both feed into the classifier as features.

### Stage 3: Motif Assignment & Feature Extraction

**For each clip** with spatial embeddings `{z_1, ..., z_15}` and global embedding `z_global`:

#### 3a. Global Motif Features

Soft membership from HDBSCAN's `membership_vector()`:
```
m_t[k] = P(global_motif_k | z_t)    # from HDBSCAN soft clustering
```
Note: unlike the temperature-scaled softmax in the original plan, this uses HDBSCAN's native probabilistic membership which accounts for cluster density and shape — more principled than distance-based softmax.

Aggregate over the 15 spatial positions per clip:
- **Motif histogram**: `h_k = Σ_t m_t[k]` → `(K_global,)` — total motif presence
- **Max activation**: `max_t m_t[k]` → `(K_global,)` — peak motif strength
- **Temporal spread**: `std_t(m_t[k])` → `(K_global,)` — sustained vs. transient
- **Noise fraction**: `Σ_t P(noise | z_t) / 15` → scalar — how much of the clip is background

#### 3b. Within-Species Sub-Cluster Features

For each candidate species `s`, compute the likelihood of the clip's global embedding under each of species s's GMM sub-clusters:
```
l_s[j] = P(z_global | sub-cluster_j of species s)    # GMM posterior
```
This produces a `(num_species, max_K_s)` matrix (padded). In practice, flatten or summarize:
- **Best sub-cluster match**: `max_j l_s[j]` per species → `(num_species,)` — "how well does the best call type of species s match this clip?"
- **Sub-cluster entropy**: `H(l_s)` per species → `(num_species,)` — low entropy = clean match to one call type; high entropy = ambiguous

#### 3c. Final Feature Vector Per Clip

```
x = [ global_embedding          (1536)
    ∥ motif_histogram            (K_global)
    ∥ max_activation             (K_global)
    ∥ temporal_spread            (K_global)
    ∥ noise_fraction             (1)
    ∥ best_subcluster_match      (num_species)
    ∥ subcluster_entropy         (num_species)
    ∥ top_perch_logits           (200) ]
```

Dimensionality: `1536 + 3*K_global + 1 + 2*num_species + 200`
- With K_global ≈ 100, num_species ≈ 200: total ≈ **2637D**
- This is manageable for an MLP — no dimensionality bottleneck

### Stage 4: Classifier Training (Initial)

**Architecture**: 2-layer MLP with dropout

```
Input(D) → Linear(512) → BatchNorm → ReLU → Dropout(0.3)
         → Linear(256) → BatchNorm → ReLU → Dropout(0.3)
         → Linear(num_species) → Sigmoid (multi-label)
```

**Training data strategy** (handles the two data sources):

1. **XC/iNat segments**: Use primary label as positive. Treat secondary labels as positive where present but do NOT treat absent secondaries as negative — use **positive-unlabeled (PU) loss** or simply mask the loss for unlabeled species columns per sample.
2. **Labeled soundscape segments**: Use as-is (multi-label, strong supervision).
3. **Unlabeled soundscapes**: Excluded from initial training — used in Stage 4b pseudo-labeling.

**Loss function**: Per-species masked BCE or focal loss
```
L = Σ_s  mask_s * BCE(ŷ_s, y_s)
```
Where `mask_s = 1` if species s is positively labeled OR if the sample comes from a fully-annotated soundscape segment (where absence = true negative). `mask_s = 0` for unknown/unlabeled species on XC/iNat secondary slots.

**Other training details**:
- Optimizer: AdamW, lr=1e-3 with cosine decay
- Validation: Stratified k-fold on **labeled soundscape data only** (most realistic eval)
- Metric: macro-averaged ROC-AUC
- Fast iteration: training takes minutes on frozen embeddings

### Stage 4b: Pseudo-Labeling

Iteratively label unlabeled data using the trained classifier, then retrain.

**Round 1 — Perch logit bootstrap**:
1. Run Perch's built-in 15K-class classifier on all unlabeled soundscape segments
2. For species in Perch's taxonomy that overlap with our target set: accept predictions above a **high confidence threshold** (e.g., top logit > 0.9) as pseudo-labels
3. Add these to training set with reduced weight (e.g., `sample_weight=0.5`)

**Round 2+ — Self-training with the motif classifier**:
1. Run the Stage 4 trained classifier on all remaining unlabeled segments
2. Apply confidence threshold per species (calibrated on validation set — use the threshold where precision > 0.9 on val)
3. Accept high-confidence predictions as pseudo-labels
4. Retrain classifier on original labels + pseudo-labels
5. Repeat until convergence (pseudo-label set stabilizes) or for a fixed number of rounds (2-3 is usually sufficient)

**Safeguards against pseudo-label noise**:
- **Threshold calibration**: Per-species thresholds, not global — rare species need different thresholds than common ones
- **Mixup regularization**: During pseudo-label retraining, use mixup on embeddings to prevent overfitting to noisy labels
- **Validation firewall**: Never pseudo-label validation data. Monitor val AUC each round — stop if it drops
- **Cluster consistency check**: If a pseudo-labeled segment's motif profile is an outlier within its predicted species' typical motif distribution, flag it as suspicious and exclude

**Why pseudo-labeling matters here**:
- XC/iNat data is clean but domain-shifted (focal recordings ≠ field soundscapes)
- Labeled soundscapes are realistic but scarce
- Unlabeled soundscapes are abundant — pseudo-labeling bridges the domain gap by adding in-domain training signal

### Stage 5: CPU Deployment

1. **Install Perch 2.0 CPU variant**: Use the Kaggle CPU release with `tensorflow-cpu` — no TFLite conversion needed
2. **Precompute and ship prototypes**: `P` is just a `(K, 1536)` matrix — trivial to store
3. **Ship classifier weights**: Tiny MLP, <5MB
4. **Inference pipeline**: Perch CPU model → numpy soft assignment → MLP forward pass
5. **Optional TFLite**: If latency is critical for real-time monitoring, TFLite export still provides ~10x additional speedup

---

## 4. Slurm Job Definitions

All jobs follow the patterns established in BirdCallClassifier. Module loads:
```bash
module purge
module load cuda/12.8.0-dh6b
module load miniconda3/latest
conda activate pantanal
```

### 4.1 `scripts/setup_env.sh` (Run once)

```bash
#!/bin/bash
# Creates conda env and installs dependencies

conda create -n pantanal python=3.10 -y
conda activate pantanal

pip install git+https://github.com/google-research/perch-hoplite.git
pip install "tensorflow[and-cuda]~=2.20.0"  # GPU nodes
# For CPU-only nodes: pip install "tensorflow-cpu~=2.20.0"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install librosa soundfile h5py hdbscan umap-learn scikit-learn
pip install pandas wandb tqdm
```

### 4.2 `scripts/extract_embeddings.sh`

```bash
#!/bin/bash
#SBATCH --job-name=perch_embed
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/embed_%j.log

module purge
module load cuda/12.8.0-dh6b
module load miniconda3/latest
conda activate pantanal

AUDIO_DIR=${AUDIO_DIR:-"data/segments"}
OUTPUT_H5=${OUTPUT_H5:-"data/embeddings.h5"}

python src/extract_embeddings.py \
    --audio_dir "$AUDIO_DIR" \
    --output "$OUTPUT_H5" \
    --batch_size 64 \
    --num_workers 8
```

### 4.3 `scripts/cluster.sh`

```bash
#!/bin/bash
#SBATCH --job-name=motif_cluster
#SBATCH --partition=batch          # CPU partition — no GPU needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                 # HDBSCAN on large embedding matrices is memory-hungry
#SBATCH --time=4:00:00
#SBATCH --output=logs/cluster_%j.log

module purge
module load miniconda3/latest
conda activate pantanal

EMBEDDING_H5=${EMBEDDING_H5:-"data/embeddings.h5"}
OUTPUT_DIR=${OUTPUT_DIR:-"data/prototypes"}

echo "=== Level 1: Global motif discovery (HDBSCAN) ==="
python src/cluster_global.py \
    --embeddings "$EMBEDDING_H5" \
    --output_dir "$OUTPUT_DIR" \
    --umap_dim 64 \
    --subsample_n ${SUBSAMPLE_N:-200000} \
    --min_cluster_size ${MIN_CLUSTER_SIZE:-50} \
    --min_samples ${MIN_SAMPLES:-10}

echo "=== Level 2: Within-species sub-clustering (GMM) ==="
python src/cluster_within_species.py \
    --embeddings "$EMBEDDING_H5" \
    --labels data/labels.csv \
    --output_dir "$OUTPUT_DIR" \
    --pca_dim 64 \
    --max_components ${MAX_COMPONENTS:-10}
```

### 4.4 `scripts/train_classifier.sh`

```bash
#!/bin/bash
#SBATCH --job-name=motif_classify
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/train_%j.log

module purge
module load cuda/12.8.0-dh6b
module load miniconda3/latest
conda activate pantanal

FOLD=${FOLD:-0}

python src/train_classifier.py \
    --embeddings data/embeddings.h5 \
    --prototypes data/prototypes/ \
    --labels data/labels.csv \
    --fold "$FOLD" \
    --epochs ${EPOCHS:-50} \
    --lr ${LR:-1e-3} \
    --wandb_project pantanal-monitoring
```

### 4.5 `scripts/pseudo_label.sh`

```bash
#!/bin/bash
#SBATCH --job-name=pseudo_label
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/pseudo_%j.log

module purge
module load cuda/12.8.0-dh6b
module load miniconda3/latest
conda activate pantanal

ROUND=${ROUND:-1}
CHECKPOINT=${CHECKPOINT:-""}
THRESHOLD=${THRESHOLD:-0.9}

python src/pseudo_label.py \
    --embeddings data/embeddings.h5 \
    --prototypes data/prototypes/ \
    --labels data/labels.csv \
    --unlabeled_meta data/unlabeled_segments.csv \
    --round "$ROUND" \
    --checkpoint "$CHECKPOINT" \
    --threshold "$THRESHOLD" \
    --output data/pseudo_labels_r${ROUND}.csv

# If round > 1, retrain with pseudo-labels
if [ "$ROUND" -gt 1 ]; then
    python src/train_classifier.py \
        --embeddings data/embeddings.h5 \
        --prototypes data/prototypes/ \
        --labels data/labels.csv \
        --pseudo_labels data/pseudo_labels_r${ROUND}.csv \
        --pseudo_weight ${PSEUDO_WEIGHT:-0.5} \
        --fold ${FOLD:-0} \
        --epochs ${EPOCHS:-50} \
        --wandb_project pantanal-monitoring \
        --run_name "pseudo_r${ROUND}_${SLURM_JOB_ID}"
fi
```

### 4.6 `scripts/run_pipeline.sh` (Orchestrator)

```bash
#!/bin/bash
# End-to-end pipeline orchestrator
# Usage: bash scripts/run_pipeline.sh [--skip-pseudo]

set -euo pipefail
SKIP_PSEUDO=${1:-""}
NUM_PSEUDO_ROUNDS=${NUM_PSEUDO_ROUNDS:-2}

echo "=== Stage 0: Data Preparation ==="
python src/prepare_data.py --raw_dir data/raw --output_dir data/segments

echo "=== Stage 1: Embedding Extraction ==="
EMBED_JOB=$(sbatch --parsable scripts/extract_embeddings.sh)
echo "Submitted embedding job: $EMBED_JOB"

echo "=== Stage 2: Clustering (depends on Stage 1) ==="
CLUSTER_JOB=$(sbatch --parsable --dependency=afterok:$EMBED_JOB scripts/cluster.sh)
echo "Submitted clustering job: $CLUSTER_JOB"

echo "=== Stage 4: Initial Classifier Training (depends on Stage 2) ==="
LAST_TRAIN_JOBS=""
for FOLD in 0 1 2 3 4; do
    TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$CLUSTER_JOB \
        --export=ALL,FOLD=$FOLD scripts/train_classifier.sh)
    LAST_TRAIN_JOBS="${LAST_TRAIN_JOBS}:${TRAIN_JOB}"
    echo "Submitted training fold $FOLD: $TRAIN_JOB"
done

if [ "$SKIP_PSEUDO" != "--skip-pseudo" ]; then
    echo "=== Stage 4b: Pseudo-Labeling ==="

    # Round 1: Perch logit bootstrap (no checkpoint needed)
    PL_JOB=$(sbatch --parsable --dependency=afterok${LAST_TRAIN_JOBS} \
        --export=ALL,ROUND=1,THRESHOLD=0.9 scripts/pseudo_label.sh)
    echo "Submitted pseudo-label round 1: $PL_JOB"

    # Round 2+: self-training with retrain
    PREV_JOB=$PL_JOB
    for ROUND in $(seq 2 $NUM_PSEUDO_ROUNDS); do
        for FOLD in 0 1 2 3 4; do
            PL_JOB=$(sbatch --parsable --dependency=afterok:$PREV_JOB \
                --export=ALL,ROUND=$ROUND,FOLD=$FOLD,THRESHOLD=0.85 \
                scripts/pseudo_label.sh)
            echo "Submitted pseudo-label round $ROUND fold $FOLD: $PL_JOB"
        done
        PREV_JOB=$PL_JOB
    done
fi

echo "Pipeline submitted. Monitor with: squeue -u $USER"
```

Key pattern: **`--dependency=afterok:<jobid>`** chains stages sequentially while allowing folds to run in parallel.

---

## 5. Critical Evaluation of the README Architecture

### What's Good

1. **Local embeddings + global clustering** is a sound framework. Decoupling motif discovery from classification means the system can find structure the labels don't describe (e.g., rain, wind, anthropogenic noise).
2. **Soft assignment** is the right call for bioacoustics. Overlapping species calls are the norm in Pantanal recordings — hard assignment would destroy this signal.
3. **Interpretability via motif heatmaps** is genuinely useful for ecologists. This is a differentiator over black-box classifiers.

### Issues and Recommended Changes

#### Issue 1: Custom Encoder is Unnecessary Work
**README says**: Build a fully convolutional encoder from scratch, optionally using a foundation model as teacher.
**Problem**: This is months of work to produce something worse than Perch 2.0, which was trained on millions of recordings across 15K species. Rolling your own encoder only makes sense if Perch's spatial embeddings are insufficient — and they won't be, given Perch 2.0 already covers frogs and insects explicitly.
**Recommendation**: Use Perch 2.0 as a frozen encoder. Invest the saved time in clustering quality and downstream classifier design.

#### Issue 2: Multi-View Spectrograms Add Complexity Without Clear Benefit
**README says**: Stack log-mel, PCEN, and frequency-boosted variants as channels.
**Problem**: Perch already uses PCEN internally. Adding more views means either (a) building a custom encoder that accepts multi-channel input (see Issue 1), or (b) running Perch multiple times on different inputs, which multiplies compute without clear evidence of benefit.
**Recommendation**: Drop multi-view preprocessing. If specific frequency ranges matter (e.g., low-frequency frogs vs. high-frequency insects), use Perch's **spatial embeddings** which already encode frequency-band information across the 3 frequency positions.

#### Issue 3: Prototype Decomposition (NMF-like) is Fragile
**README says**: Option 3 is `z_t ≈ Σ_k w_t[k] · p_k` — decompose embeddings as a weighted sum of prototypes.
**Problem**: This assumes prototypes span the embedding space linearly, which is unlikely for a nonlinear encoder like EfficientNet. Reconstruction error will be high, and optimizing the decomposition adds a non-convex step that can diverge.
**Recommendation**: Start with soft clustering (Option 1) only. NMF-like decomposition is a research tangent — revisit only after the core pipeline is validated.

#### Issue 4: HDBSCAN Scalability
**README says**: HDBSCAN recommended.
**Problem**: HDBSCAN has O(n²) memory for the mutual reachability graph. With 100K segments × 15 spatial positions = 1.5M embeddings, this requires ~18TB of pairwise distances at float32.
**Status**: ✅ Addressed in Stage 2 — subsample ~200K embeddings for HDBSCAN, use `approximate_predict()` for the rest. Alternative two-tier approach (mini-batch k-means → HDBSCAN on centroids) available as fallback.

#### Issue 5: No Handling of Temporal Context
**README says**: Local embeddings are per-patch, aggregated via histograms.
**Problem**: A 5-second Perch window captures one moment. Many species are identifiable by **temporal patterns across windows** (e.g., repeated call bouts over 30s, dawn chorus timing). Bag-of-motifs discards this.
**Recommendation**: Add a lightweight temporal model over sequential windows:
  - Option A: Sliding window of motif histograms → 1D CNN or GRU
  - Option B: Attention over sequential global embeddings (last 6 windows = 30s context)
  - This is cheap and can dramatically help with species that have distinctive call cadences.

#### Issue 6: Missing Active Learning / Annotation Strategy
**README says**: "Labels are used to map motifs → species (not to define motifs)"
**Status**: ✅ Addressed in Stage 4b — pseudo-labeling pipeline uses Perch logits for bootstrap (Round 1) then self-training (Round 2+). Cluster-level label propagation is also possible: if a global motif cluster is dominated by one species' pseudo-labels, propagate to unlabeled members.

#### Issue 7: Within-Species Structure
**README says**: Analyze residuals within clusters to find sub-structure.
**Original concern**: Premature for v1.
**Revised**: Now incorporated as Level 2 clustering (GMM per species). This is no longer a research tangent — within-species sub-clusters directly feed the classifier as features (`best_subcluster_match`, `subcluster_entropy`). The GMM approach is lightweight and automatic (BIC selects K), so it doesn't add significant implementation burden.

---

## 6. Recommended Priority Order

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| **P0** | Environment setup + Perch 2.0 installation on Easley | 1 day | Unblocks everything |
| **P0** | Data preparation pipeline (segmentation, QC) | 2-3 days | Unblocks embedding |
| **P1** | Embedding extraction (Stage 1) | 1 day | Core data asset |
| **P1** | Label provenance pipeline (masked loss setup) | 1 day | Correct training signal from mixed sources |
| **P2** | Global clustering (HDBSCAN + subsample) | 2-3 days | Core motif discovery |
| **P2** | Within-species sub-clustering (GMM + BIC) | 1-2 days | Fine-grained call type features |
| **P2** | Motif feature extraction (Stage 3) | 1 day | Feeds classifier |
| **P3** | Initial classifier training (Stage 4) | 1-2 days | End-to-end evaluation |
| **P3** | Evaluation framework (metrics, visualization) | 1-2 days | Know if it works |
| **P3** | Pseudo-labeling pipeline (Stage 4b) | 2-3 days | Bridges domain gap, uses unlabeled soundscapes |
| **P4** | Temporal context model | 2-3 days | Performance boost |
| **P4** | CPU inference validation (Kaggle CPU release) | 0.5 day | Deployment readiness |
| **P5** | Interpretability / motif visualization | 2-3 days | Ecologist-facing output |
| **P5** | NMF-like prototype decomposition | Research | Defer unless soft assignment proves insufficient |

---

## 7. Directory Structure

```
PantanalSpeciesMonitoring/
├── README.md
├── GAMEPLAN.md              ← this document
├── environment.yml
├── requirements.txt
├── scripts/
│   ├── setup_env.sh
│   ├── extract_embeddings.sh
│   ├── cluster.sh
│   ├── train_classifier.sh
│   ├── pseudo_label.sh
│   └── run_pipeline.sh
├── src/
│   ├── prepare_data.py
│   ├── extract_embeddings.py
│   ├── cluster_global.py
│   ├── cluster_within_species.py
│   ├── build_features.py
│   ├── train_classifier.py
│   ├── pseudo_label.py
│   ├── inference.py
│   └── utils/
│       ├── audio.py
│       └── viz.py
├── data/
│   ├── raw/                 # original recordings
│   ├── segments/            # 5s WAV segments
│   ├── embeddings.h5        # Perch embeddings
│   ├── prototypes/          # cluster outputs
│   ├── labels.csv           # ground truth (with source & mask columns)
│   ├── unlabeled_segments.csv
│   └── pseudo_labels_r*.csv # pseudo-labels per round
├── checkpoints/
├── logs/
└── notebooks/
    ├── 01_eda.ipynb
    ├── 02_cluster_viz.ipynb
    └── 03_motif_explorer.ipynb
```

---

## 8. Open Questions for Discussion

1. **How much labeled soundscape data do we have vs. unlabeled?** Ratio determines how aggressively to pseudo-label.
2. **What fraction of target Pantanal species are in Perch's 15K taxonomy?** Determines Round 1 pseudo-label coverage. Species outside Perch's taxonomy won't get Perch-logit pseudo-labels — they rely entirely on Round 2+ self-training.
3. **What's the target deployment environment?** (Raspberry Pi? Edge server? Cloud?) This affects whether CPU Perch is sufficient or TFLite is needed.
4. **Real-time vs batch inference?** Continuous monitoring requires streaming; periodic surveys can batch-process.
5. **Which species are priority targets?** We may want species-specific thresholds and per-species pseudo-label quality gates.
6. **How noisy are the XC/iNat secondary labels?** If very noisy, we may want to discard them entirely and treat XC/iNat as single-label only.

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Perch 2.0 TF 2.20 incompatible with Easley CUDA | Medium | High | Test immediately in P0; fallback to Perch v1 (TF 2.x stable) with 1280-D embeddings |
| HDBSCAN OOM on full embedding set | High | Medium | Subsample strategy (Section 5, Issue 4) |
| Insufficient labeled data for supervised training | Medium | High | Staged pseudo-labeling: Perch logits (Round 1) → self-training (Round 2+) |
| Pseudo-label noise degrades classifier | Medium | Medium | Per-species thresholds, val firewall, cluster consistency check, mixup regularization |
| XC/iNat domain shift to soundscapes | High | High | Pseudo-labeling on in-domain unlabeled soundscapes bridges the gap; masked loss avoids false negatives from incomplete XC secondary labels |
| Pantanal species not in Perch's 15K taxonomy | Low | Medium | Embeddings still transfer well to unseen species (demonstrated for marine mammals); clustering handles novel species |
| CPU Perch release missing spatial embeddings | Low | Medium | Verify spatial output parity with GPU variant early; fallback to global embeddings only if needed |

---

*Ready for review. Once approved, I'll scaffold the repo and start with P0 tasks.*
