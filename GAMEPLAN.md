# Pantanal Species Monitoring — Implementation Gameplan

**Date**: 2026-03-28 (updated)
**Status**: IN PROGRESS

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
- **Foundation model as teacher** → Perch is the teacher model for representation learning, but we still distill a lightweight student embedder for deployment-constrained inference.

### 1.3 What Perch Does NOT Replace

- **Global clustering (Section C)** — still needed, applied to Perch spatial embeddings
- **Soft assignment (Section D)** — still needed, core novelty of the approach
- **Motif features (Section E)** — still needed
- **Classifier (Section F)** — still needed, but now much simpler (MLP on frozen embeddings + motif features)

### 1.4 GPU vs CPU Strategy

| Phase | Hardware | Rationale |
|-------|----------|-----------|
| Teacher embedding extraction | **GPU** (L40S) | Batch-process all audio through Perch 2.0. One-time cost. |
| Student distillation training | **GPU** (L40S) | Train a fast PyTorch embedder to match Perch embeddings on diverse audio. |
| Clustering | **CPU** | HDBSCAN/k-means on extracted embeddings. Memory-bound, not GPU-bound. |
| Classifier training | **GPU** (L40S) | Train MLP head on embeddings + motif features. Fast. |
| Inference (deployment) | **CPU** | Use distilled student embeddings + precomputed prototypes + lightweight classifier. |

**Why distillation is required for viable inference**: The Kaggle CPU Perch 2.0 release was benchmarked and **does not fit within the competition time budget**. Perch's EfficientNet-B3 backbone (~12M params + 91M logit head) is too heavy for batch CPU inference at competition scale — it consumes the entire ~90 min time budget on its own, leaving zero time for prototype assignment, classifier inference, I/O, and safety margin. Distillation moves the heavy Perch compute offline (HPC/GPU, one-time cost) and keeps deployment-time inference fast and deterministic on CPU.

**Why a domain-specialized student is better — not just acceptable**: Perch's 1536-D embedding space is optimized for global taxonomic diversity across ~15,000 species worldwide. Our task only requires discriminating 234 Pantanal species. A student trained to mimic Perch's output on Pantanal-domain audio learns a **compressed, domain-specialized** embedding space — dimensions irrelevant to Pantanal bioacoustics are discarded. The student can be substantially smaller than Perch with no loss of discriminative signal for our specific task. This is a genuine advantage, not a compromise.

**Why add more external data before distillation**: the student only generalizes to acoustic conditions seen during teacher-student training. Adding tropical/soundscape-heavy audio (BirdCLEF unlabeled soundscapes + filtered iNat/BirdSet subsets) improves robustness to domain shift in noisy, polyphonic recordings.

**Deduplication is mandatory**: cross-source duplication (e.g., iNaturalist overlap across datasets) can leak near-identical clips into multiple splits, distort metrics, and waste storage/compute. We deduplicate by source ID, filename keys, and content hash.

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
│ STAGE 0b: Distill Data Preparation  [CPU+I/O]              ✅    │
│   distill_manifest.csv → dedup vs segments.csv (sha1+basename)   │
│   → segment into 5s chunks → label-match via taxonomy             │
│   → 135/234 species covered; unmatched = unlabeled (still useful) │
│   Output: outputs/distill_segments.csv (same 11-col schema)       │
├──────────────────────────────────────────────────────────────────┤
│ STAGE 1b: Scrape External Corpus  [CPU+I/O]                ✅    │
│   iNat Sounds 2024 (Pantanal bbox): 805 files extracted           │
│   BirdSet PER (Peru, Neotropical): scrape in progress             │
│   Dedup: sha1 + basename across both sources + across runs        │
│   Output: data/distill_manifest.csv, data/distill_audio/          │
├──────────────────────────────────────────────────────────────────┤
│ STAGE 1c: Embed Distill Corpus with Perch (Teacher) [GPU]         │
│   distill_segments.csv → Perch embeddings/logits                  │
│   Same extract_embeddings.py, overriding segments/h5 config keys  │
│   Output: outputs/embeddings/distill_embeddings.h5                │
├──────────────────────────────────────────────────────────────────┤
│ STAGE 1d: Distill Student Embedder  [GPU]                         │
│   EfficientNet-B1-BirdSet backbone + projection heads             │
│   student(audio) ≈ Perch(audio) in embedding space                │
│   Loss: cosine(global) + cosine(spatial) + 0.15×MSE(logits)       │
│   Output: CPU-deployable student (~58ms/clip on CPU)              │
├──────────────────────────────────────────────────────────────────┤
│ STAGE 2: Motif Discovery  [CPU]                                  │
│   primary + distill spatial embeddings (--with-distill flag)     │
│   → HDBSCAN → prototypes P = {p_1..p_K}                          │
│   → per-species GMMs enriched with distill labeled segments       │
│   Output: prototype vectors + cluster metadata + species GMMs    │
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
│   new audio → distilled student → embeddings → assign prototypes │
│   → motif features → classifier → species predictions            │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Implementation Stages (Detailed)

### Stage 0: Data Preparation ✅

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

**Implemented** (`src/prepare_data.py`):
- Segments XC/iNat focal recordings and soundscapes into 5s windows
- Silence filter (frame-level RMS energy threshold, configurable)
- Outputs `outputs/segments.csv` with 11-column schema: `segment_id, source_file, start_sec, end_sec, source_type, primary_label, secondary_labels, is_labeled, label_quality`
- **Current status**: 358,455 segments (~348K labeled, ~10K unlabeled soundscapes)

**Output**: `outputs/segments.csv`

---

### Stage 0b: Distill Data Preparation ✅

Parallel pipeline for external distillation corpus (iNat Sounds 2024 + BirdSet). Keeps distill data in a separate CSV to avoid polluting the primary training split structure.

**Implemented** (`src/prepare_distill_data.py`):

**Deduplication** (two layers):
1. **sha1 hash** — exact content duplicates across runs are skipped
2. **Audio file basename** — catches any overlap between distill audio and files already in `segments.csv` train set

**Label matching** — joins `distill_manifest.csv` `species_name` (scientific name) against `taxonomy.csv` via case-insensitive string match:
- 135/234 target species have matching distill audio (all labeled `strong_primary`)
- Remaining 99 species have no distill coverage → those files segmented as `is_labeled=False` (still useful for HDBSCAN motif discovery)

**Segmentation**: identical 5s logic as Stage 0 (reuses `segment_file()` and `check_energy()` from `prepare_data.py`)

**Output**: `outputs/distill_segments.csv` — same 11-column schema as `segments.csv`, `segment_id` prefixed with `distill/` for traceability

### Stage 1b: External Corpus Scrape ✅

**Implemented** (`src/scrape_distill_data.py`, `scripts/scrape_distill_data.sh`):

| Source | Config | Target | Status |
|--------|--------|--------|--------|
| iNat Sounds 2024 | Pantanal bbox (`-25,-60,-10,-45`), aves/amphibia/insecta/mammalia/reptilia, ≥2s | 8,000 files | ✅ 805 files extracted (full candidate set) |
| BirdSet PER | Peru config, train split | 3,000 files | 🔄 Re-running with HF token (prior run got 1 file — unauthenticated rate limit) |

**Dedup** (within scraper, before writing): source ID + content sha1 + audio basename. Manifest tracks all three.

**Section-level skip logic** in `scrape_distill_data.sh`: iNat section skipped if manifest already has `inat,` rows; BirdSet skipped if count ≥ target — safe to resubmit.

**Bugs fixed during scrape runs**:
- iNat: `file_name` in metadata already includes `train/` prefix → was being double-prefixed as key → 0 extractions fixed
- iNat: `os.replace()` fails across filesystems (`/tmp` → `/users`) → switched to `shutil.move()`
- BirdSet: missing `trust_remote_code=True` → added
- BirdSet: missing `librosa` in isolated venv → added to install

**Output**: `data/distill_manifest.csv`, `data/distill_audio/{inat_sounds_2024,birdset}/`

---

### Stage 1: Embedding Extraction ✅

**Tool**: `perch-hoplite` with `perch_v2` model, implemented in `src/extract_embeddings.py`

**Storage** (`outputs/embeddings/embeddings.h5`):
- `global_embeddings`: `(N, 1536)` float32 — mean-pooled from spatial
- `spatial_embeddings`: `(N, 5, 3, 1536)` float32
- `logit_values`: `(N, 200)` float16 — top-K Perch logits
- `logit_indices`: `(N, 200)` int16 — class indices for top-K
- `written`: `(N,)` bool — resumability flag
- `segment_ids`: string array

**Resumable**: checks `written` array, skips already-embedded segments on restart.

**Estimated size**: ~2.4 GB for 358K segments

---

### Stage 1c: Embed Distill Corpus ⬜

**Implemented** (`scripts/extract_distill_embeddings.sh`):
- Runs Stage 0b (`prepare_distill_data.py`) if `distill_segments.csv` doesn't exist yet
- Runs Stage 1 (`extract_embeddings.py`) with config overrides:
  - `outputs.segments_csv=outputs/distill_segments.csv`
  - `outputs.embeddings_h5=outputs/embeddings/distill_embeddings.h5`
- No changes to `extract_embeddings.py` — override via `--set` flags

**To run**:
```bash
sbatch scripts/extract_distill_embeddings.sh
```

**Output**: `outputs/distill_segments.csv`, `outputs/embeddings/distill_embeddings.h5`

### Stage 2: Motif Discovery (Two-Level Clustering) ✅ (primary) / ⬜ (with-distill)

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

#### Distill Merge (`--with-distill`)

When `distill_embeddings.h5` is present, `cluster.sh` automatically passes `--with-distill`:

**Level 1 (HDBSCAN)**: subsample budget split proportionally between primary and distill embeddings. Distill audio covers Pantanal + Peru biomes — adds acoustic diversity to motif prototypes without displacing primary data.

**Level 2 (GMMs)**: for each species, distill labeled segments are concatenated with primary labeled segments before PCA + BIC fitting. Species with ≥1 distill file benefit from richer embedding pools → more stable GMM components. 135/234 target species have distill coverage.

**Labeled distill segments added to training data**: any distill segment with a matched taxonomy label (`is_labeled=True`) is treated as `strong_primary` — same loss mask as XC/iNat. This directly expands the labeled training set for 135 species, most of which are data-scarce in the primary corpus.

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

### Stage 1d: Student Embedder Distillation

**Motivation**: Perch 2.0 CPU inference does not fit within the Kaggle competition time budget. A distilled student embedder replicates Perch's embedding geometry for Pantanal-domain audio at a fraction of the compute cost. Because the student is trained specifically on the Pantanal audio distribution, it is also a *better* encoder for this task — not just a faster one.

#### Student Architecture

**Backbone**: `DBD-research-group/EfficientNet-B1-BirdSet-XCL` (19.8M params, same family used in BirdCallClassifier). Key advantages over training from scratch:
- Already pretrained on ~500K bird audio clips (BirdSet-XCL corpus)
- Produces well-structured audio embeddings out of the box
- ~2× smaller than Perch's EfficientNet-B3 backbone
- Runs a 5s clip in ~40–60ms on CPU

**Output heads** (added on top of BirdSet backbone):
```
EfficientNet-B1 backbone → feature map
    → Global pool → Linear(1536)           # global embedding head
    → Adaptive pool (5,3) → Linear(1536)  # spatial embedding head (per cell)
```

The spatial head pools the feature map to a `(5, 3)` grid matching Perch's spatial layout, then projects each cell to 1536-D. This preserves the motif assignment pipeline downstream without any changes to `build_features.py`.

#### Distillation Targets

All targets are pre-computed in `outputs/embeddings.h5` — no Perch forward passes needed at student training time:

| Target | Shape | Loss | Weight |
|--------|-------|------|--------|
| Global embedding | (1536,) | Cosine loss `1 - cos_sim` | 1.0 |
| Spatial embeddings | (5, 3, 1536) | Cosine loss per cell, mean | 1.0 |
| Top-200 Perch logits | (200,) | MSE | 0.15 |

**Why cosine loss for embeddings**: prototype assignment uses cosine similarity, so we care about *direction* in embedding space, not magnitude. MSE on raw values is the wrong objective — it penalizes magnitude differences that have no effect on downstream assignment.

**Why include logit distillation**: the top-200 Perch logits carry taxonomic signal (soft species labels) that helps the student understand Pantanal-relevant structure. Weight is low (0.15) since it's auxiliary to the embedding objective. This mirrors the distillation setup already running in BirdCallClassifier (job 321796, `weight=0.15, temperature=2.0`).

#### Training Setup

```python
# Distillation loss
L = cosine_loss(student_global, teacher_global)
  + cosine_loss(student_spatial, teacher_spatial)   # mean over 15 cells
  + 0.15 * mse_loss(student_logits, teacher_logits)

# where cosine_loss(a, b) = mean(1 - F.cosine_similarity(a, b, dim=-1))
```

- **Optimizer**: AdamW, lr=1e-4 with cosine decay (backbone lr × 0.1 for frozen layers)
- **Input**: mel spectrogram matching Perch's frontend — 128 mel bins, 32kHz, hop=10ms, window=25ms
- **Data**: all 358K segments from `outputs/segments.csv` with valid embeddings in HDF5
- **Epochs**: 20–30 (convergence is fast — targets are fixed teacher outputs, no label noise)
- **Hardware**: GPU (L40S), ~2–4h

#### Validation

Two checks before accepting the student:

1. **Embedding alignment**: cosine similarity between student and teacher embeddings on a held-out set. Target: mean cosine sim > 0.85.
2. **Downstream proxy**: run the full feature pipeline (prototype assignment → MLP) with student embeddings instead of Perch. Val macro-AUC should be within ~2pp of the teacher-embedding baseline.

#### Inference pipeline with student (final deployed form)

```
Audio clip (5s @ 32kHz)
  → mel spectrogram (128 bins, 32kHz — precomputed or on-the-fly)
  → Student EfficientNet-B1 → global_emb (1536,), spatial_emb (5,3,1536)
  → cosine sim to frozen prototypes (2048, 1536)   [matrix multiply only]
  → feature vector (same build_features.py logic, unchanged)
  → MotifClassifier MLP (tiny, <5MB)
  → species logits
```

**Nothing downstream of the student changes.** The prototype matrix, feature construction logic, and MLP are all identical to the teacher-embedding pipeline. The student is a drop-in replacement for Perch at inference time.

#### Estimated CPU inference time (Kaggle)

| Component | Time per 5s clip | Notes |
|-----------|-----------------|-------|
| Mel spectrogram | ~2ms | librosa/numpy |
| Student EfficientNet-B1 | ~50ms | PyTorch CPU, bfloat16 |
| Prototype assignment | ~5ms | (2048, 1536) matmul |
| Feature concat + MLP | ~1ms | tiny MLP |
| **Total** | **~58ms/clip** | |

A 60-min soundscape at 5s stride = 720 clips × 58ms ≈ **42 seconds**. Well within the ~90 min budget, leaving ample margin for I/O and overhead.

### Stage 5: CPU Deployment

1. **Ship student weights**: PyTorch checkpoint, ~80MB (EfficientNet-B1)
2. **Ship prototypes**: `global_prototypes.npz` — `(2048, 1536)` float32 matrix, ~12MB
3. **Ship GMMs**: `species_gmms.pkl` — per-species PCA + GMM models, ~few MB
4. **Ship classifier**: `MotifClassifier` MLP weights, <5MB
5. **Inference pipeline**: student → prototype assignment → feature vector → MLP → species logits (see Stage 1d above)
6. **No TensorFlow at inference**: student is pure PyTorch, no TF dependency on Kaggle

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

### 4.2 `scripts/extract_embeddings.sh` ✅

Implemented. GPU job (L40s, 12h). Reads `outputs/segments.csv` → writes `outputs/embeddings/embeddings.h5`. Fully resumable via `written` flag in HDF5.

### 4.2b `scripts/extract_distill_embeddings.sh` ✅

New script. Same GPU setup as `extract_embeddings.sh`. Runs Stage 0b (`prepare_distill_data.py`) if `distill_segments.csv` is missing, then runs `extract_embeddings.py` with config key overrides to write `outputs/embeddings/distill_embeddings.h5`.

```bash
sbatch scripts/extract_distill_embeddings.sh
```

### 4.3 `scripts/cluster.sh` ✅

Implemented. CPU job (16 cores, 128GB, 6h). Runs Stage 2 (`cluster.py`) + Stage 3 (`build_features.py`) sequentially.

**Auto-detects distill embeddings**: if `outputs/embeddings/distill_embeddings.h5` exists, automatically passes `--with-distill --distill-h5 <path>` to `cluster.py`. No manual flag needed.

### 4.4 `scripts/scrape_distill_data.sh` ✅

CPU job (4 cores, 32GB, 24h). Scrapes iNat + BirdSet audio. Section-level skip logic prevents re-running completed sources on resubmit.

```bash
sbatch scripts/scrape_distill_data.sh
# Override defaults:
INAT_MAX_FILES=1000 BIRDSET_MAX_FILES=500 sbatch scripts/scrape_distill_data.sh
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

| Priority | Task | Effort | Impact | Status |
|----------|------|--------|--------|--------|
| **P0** | Environment setup + Perch 2.0 installation on Easley | 1 day | Unblocks everything | ✅ Done |
| **P0** | Data preparation pipeline (segmentation, QC) | 2-3 days | Unblocks embedding | ✅ Done |
| **P1** | External corpus scrape (iNat + BirdSet) | 1-2 days | Distill data asset | ✅ iNat done / 🔄 BirdSet in progress |
| **P1** | Distill data preparation (Stage 0b) | 1 day | Dedup + segment distill corpus | ✅ Done |
| **P1** | Embedding extraction (Stage 1) | 1 day | Core data asset | ✅ Done |
| **P1** | Distill embedding extraction (Stage 1c) | 0.5 day | Teacher targets for distillation + richer clustering | ⬜ Ready to run |
| **P1** | Label provenance pipeline (masked loss setup) | 1 day | Correct training signal from mixed sources | ✅ Done |
| **P1** | Student embedder distillation (Stage 1d) | 2-3 days | Required for deployment — Perch CPU too slow | ⬜ Pending |
| **P2** | Global clustering (HDBSCAN + subsample) | 2-3 days | Core motif discovery | ✅ Done (primary) / ⬜ with-distill pending 1c |
| **P2** | Within-species sub-clustering (GMM + BIC) | 1-2 days | Fine-grained call type features | ✅ Done (primary) / ⬜ with-distill pending 1c |
| **P2** | Motif feature extraction (Stage 3) | 1 day | Feeds classifier | ✅ Done |
| **P3** | Initial classifier training (Stage 4) | 1-2 days | End-to-end evaluation | ⬜ Pending |
| **P3** | Evaluation framework (metrics, visualization) | 1-2 days | Know if it works | ⬜ Pending |
| **P3** | Pseudo-labeling pipeline (Stage 4b) | 2-3 days | Bridges domain gap, uses unlabeled soundscapes | ⬜ Pending |
| **P4** | Temporal context model | 2-3 days | Performance boost | ⬜ Defer |
| **P4** | Student CPU inference validation (timing on Kaggle) | 0.5 day | Deployment readiness | ⬜ Pending 1d |
| **P5** | Interpretability / motif visualization | 2-3 days | Ecologist-facing output | ⬜ Defer |
| **P5** | NMF-like prototype decomposition | Research | Defer unless soft assignment proves insufficient | ⬜ Defer |

---

## 7. Directory Structure

```
PantanalSpeciesMonitoring/
├── README.md
├── GAMEPLAN.md              ← this document
├── environment.yml
├── configs/
│   └── default.yaml         # all stage configs + output paths
├── scripts/
│   ├── setup_env.sh
│   ├── scrape_distill_data.sh        ✅ iNat + BirdSet scraper
│   ├── extract_embeddings.sh         ✅ primary Perch embed (GPU)
│   ├── extract_distill_embeddings.sh ✅ distill Stage 0b + 1c (GPU)
│   ├── cluster.sh                    ✅ Stage 2+3, auto --with-distill
│   ├── train_classifier.sh
│   ├── pseudo_label.sh
│   └── run_pipeline.sh
├── src/
│   ├── prepare_data.py               ✅ Stage 0: segment XC/iNat/soundscapes
│   ├── prepare_distill_data.py       ✅ Stage 0b: segment + dedup distill corpus
│   ├── extract_embeddings.py         ✅ Stage 1/1c: Perch embed (config-driven)
│   ├── scrape_distill_data.py        ✅ Stage 1b: iNat + BirdSet scraper
│   ├── cluster.py                    ✅ Stage 2: HDBSCAN + GMM (--with-distill)
│   ├── build_features.py             ✅ Stage 3: motif features
│   ├── train_classifier.py           Stage 4: MLP classifier
│   ├── pseudo_label.py               Stage 4b: pseudo-labeling
│   ├── config.py                     ✅ YAML config loader with --set overrides
│   └── utils.py                      ✅ shared audio/logging utilities
├── data/
│   ├── train_audio/         # XC/iNat focal recordings
│   ├── train_soundscapes/   # field soundscape recordings
│   ├── train.csv            # XC/iNat metadata
│   ├── train_soundscapes_labels.csv
│   ├── taxonomy.csv         # 234-species label set
│   ├── distill_manifest.csv ✅ iNat+BirdSet scrape manifest
│   └── distill_audio/       ✅ scraped audio files
│       ├── inat_sounds_2024/train/   (805 files)
│       └── birdset/PER/train/        (in progress)
└── outputs/
    ├── segments.csv                  ✅ 358K segments (primary)
    ├── distill_segments.csv          ✅ distill segments (same schema)
    ├── embeddings/
    │   ├── embeddings.h5             ✅ 2.4GB primary embeddings
    │   └── distill_embeddings.h5     ⬜ pending Stage 1c
    ├── prototypes/
    │   ├── global_prototypes.npz     ✅
    │   ├── global_hdbscan.pkl        ✅
    │   ├── global_umap.pkl           ✅
    │   ├── global_cluster_stats.csv  ✅
    │   └── species_gmms.pkl          ✅
    ├── features/
    │   └── features.h5               ✅
    ├── checkpoints/                  ⬜ Stage 4
    ├── pseudo_labels/                ⬜ Stage 4b
    └── logs/
```

---

## 8. Open Questions for Discussion

1. **How much labeled soundscape data do we have vs. unlabeled?** Ratio determines how aggressively to pseudo-label.
2. **What fraction of target Pantanal species are in Perch's 15K taxonomy?** Determines Round 1 pseudo-label coverage. Species outside Perch's taxonomy won't get Perch-logit pseudo-labels — they rely entirely on Round 2+ self-training.
3. **What's the target deployment environment?** Kaggle CPU notebook confirmed as primary target. EfficientNet-B1 student fits comfortably. For edge (Raspberry Pi / real-time monitoring), may need further compression to EfficientNet-B0 or TFLite export.
4. **Real-time vs batch inference?** Competition uses batch (full soundscape files). Continuous monitoring would need streaming — student inference is fast enough for near-real-time at 5s stride.
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
| Student embedding alignment insufficient (cosine sim < 0.85) | Medium | High | Increase distillation epochs; add hard negative mining; fall back to fine-tuning BirdSet backbone directly on species labels |
| Student CPU inference still too slow | Low | High | EfficientNet-B1 benchmarked at ~50ms/clip CPU; if needed, downgrade to EfficientNet-B0 (~25ms/clip) at minor accuracy cost |

---

## 10. Supplemental Ideas / Future Directions

### A. End-to-End Differentiable Clustering

The core pipeline treats clustering as a frozen preprocessing step — HDBSCAN discovers prototypes offline, then the classifier uses them as fixed features. This leaves performance on the table because the prototypes are not optimized for classification.

**The problem**: HDBSCAN/GMM hyperparameters (min_cluster_size, number of components) are discrete or non-differentiable, so classification loss can't directly flow back through them.

**Approaches, in order of practicality:**

#### 1) Learnable prototypes + temperature (recommended first step)
Initialize prototypes `P` from HDBSCAN medoids, then make them `nn.Parameter`. The soft assignment is already differentiable:
```
w_t[k] = softmax(-||z_t - p_k||^2 / τ_k)  →  motif features  →  classifier  →  loss
```
Gradients from classification loss flow back to both `P` (prototype positions) and per-prototype `τ_k` (assignment sharpness). This is essentially a prototypical network initialized from unsupervised clustering.

- Tight clusters (e.g., a specific bird call) will learn low `τ_k` (sharp assignment)
- Diffuse clusters (e.g., background noise) will learn high `τ_k` (soft assignment)
- Dead prototypes (never activated) can be pruned during training

#### 2) Gumbel-Softmax for hard-ish assignments
If interpretability requires near-discrete assignment (e.g., "this segment activates motifs 3, 17, and 42"), use Gumbel-Softmax with temperature annealing. Start soft for gradient flow, anneal toward hard for inference. Compatible with approach (1).

#### 3) Bayesian hyperparameter optimization (outer loop)
Treat HDBSCAN hyperparameters (min_cluster_size, min_samples, UMAP dim) as outer-loop variables optimized via Optuna or similar. Inner loop: cluster → build features → train classifier → val AUC. Not true backprop, but achieves the same goal. Expensive (full retrain per trial) but embarrassingly parallel across Slurm jobs.

#### 4) Full joint deep clustering (most ambitious)
Merge Stages 2–4 into a single end-to-end model:
```
Perch embeddings → learned prototypes (nn.Parameter) → soft assign → features → classifier
                          ↑                                                         |
                          └──────────── gradient from classification loss ───────────┘
```
K (number of prototypes) is fixed per run but can be set high and pruned. HDBSCAN provides initialization only. This is a research direction — worth exploring after the baseline pipeline is validated.

**Recommendation**: Implement (1) as a Stage 4 variant after the frozen-prototype baseline is working. If val AUC improves, adopt it. Use (3) to tune HDBSCAN hyperparameters regardless. Defer (4) unless (1) shows large gains, suggesting the prototype positions matter a lot.

**Note**: Within-species GMMs should stay frozen — they serve interpretability and the global motif features already carry most classification signal.

### B. Attention-Based Prototype Assignment

The current soft assignment uses a fixed similarity function (cosine or Euclidean distance). This assumes all dimensions of the embedding space are equally important for determining cluster membership. An attention mechanism lets the model **learn** which dimensions matter for each prototype.

#### Cross-attention over prototypes
Treat prototypes as keys, local embeddings as queries:
```
Q = z_t @ W_q          # (d_k,)   — query from local embedding
K_k = p_k @ W_k        # (d_k,)   — key from prototype k
V_k = p_k @ W_v        # (d_v,)   — value from prototype k

attn_t[k] = softmax(Q · K_k / sqrt(d_k))    # attention weight
output_t = Σ_k attn_t[k] · V_k              # attended prototype mixture
```

This has several advantages over distance-based assignment:
- **Learned relevance**: `W_q` and `W_k` can suppress irrelevant dimensions (e.g., amplitude-related features when matching tonal structure)
- **Asymmetric matching**: a prototype can "attend to" an embedding differently than the reverse — useful when prototypes represent abstract categories
- **Richer output**: the attended output `output_t` is a learned mixture of prototype values, not just a weight vector — this can be pooled and fed directly to the classifier

#### Multi-head variant
Use multiple attention heads to capture different aspects of the embedding-prototype relationship (e.g., one head for spectral shape, another for temporal pattern).

#### When to try this
After approach A1 (learnable prototypes). If learnable prototypes + temperature give a significant lift over frozen prototypes, that signals the assignment function matters a lot, and attention would be the next step. If learnable prototypes don't help, attention won't either — the bottleneck is elsewhere.

**Complexity**: Adds `W_q`, `W_k`, `W_v` matrices (~3 × 1536 × d_k parameters per head). Training cost is negligible since features are precomputed. The assignment step becomes part of the model's forward pass rather than a preprocessing step, which means it must be reimplemented in PyTorch rather than numpy — but the prototype features and MLP can be trained jointly in a single forward/backward pass.

### D. Cross-Project Student Transfer

The BirdCallClassifier project (BirdCLEF 2026) is training `EfficientNet-B1-BirdSet-XCL` with distillation (`weight=0.15, temperature=2.0`) on the same 234-class Pantanal label set. Once that model converges, its checkpoint is a strong initialization for our student embedder:

- It already speaks the right taxonomic vocabulary (234 Pantanal species)
- Distillation in Stage 1d can fine-tune it further to match Perch's *embedding geometry* specifically, rather than classification logits
- This avoids 20+ epochs of training from scratch and may yield better spatial embedding alignment

**If BirdCallClassifier achieves good val AUC**, consider using its checkpoint directly as the student and skipping Stage 1d distillation entirely — replacing the global embedding component with BirdSet embeddings and relying on the BirdSet model's feature space for prototype assignment. The embedding dimension will differ (BirdSet outputs vary by backbone), so prototype recomputation would be needed. Worth evaluating as an ablation.

### C. No-Clustering Baseline

Before investing in clustering improvements, establish a baseline: **global Perch embedding + top-K logits → MLP → species predictions**. No prototypes, no motif features. If this baseline already performs well, the marginal value of clustering is the delta above it — and optimization effort should focus wherever that delta is largest.

---

*Ready for review. Once approved, I'll scaffold the repo and start with P0 tasks.*

---

## 11. Implementation Status (as of 2026-03-28)

This section tracks what is actually built, running, and missing. The pipeline has evolved across multiple sessions — some parts were implemented before later design decisions (student distillation, cosine similarity) were finalized, so there are intentional gaps where code predates the current architecture.

### Completed ✅

| Component | File(s) | Notes |
|-----------|---------|-------|
| Config system | `src/config.py` | YAML + CLI `--set` overrides + `OVERRIDE` env var; resolves paths against PROJECT_ROOT |
| Shared utilities | `src/utils.py` | `load_audio_segment`, `build_label_map`, `parse_soundscape_labels`, `setup_logging` |
| Stage 0: Data prep | `src/prepare_data.py` | 358,454 segments, 234 species (incl. 25 insect sonotypes), label provenance tracked |
| Stage 0b: Distill data prep | `src/prepare_distill_data.py` | Segments distill audio to 5s, deduplicates against segments.csv by basename + sha1 |
| Stage 1: Perch embedding extraction | `src/extract_embeddings.py` | HDF5 with global (1536), spatial (5,3,1536), top-200 logits float16; resumable via `written` flag |
| Distill data scraping | `src/scrape_distill_data.py` | iNat Sounds 2024 (bbox-filtered, Pantanal region) + BirdSet PER; resumable; sha1 dedup; isolated venv for BirdSet |
| Stage 2: Global clustering | `src/cluster.py` (Level 1) | UMAP 64D → HDBSCAN → **2048 prototypes**; medoids stored in original 1536-D space; `prediction_data=True` for future approximate_predict |
| Stage 2: Within-species GMMs | `src/cluster.py` (Level 2) | 234 species, BIC model selection, mean K=3.1 components, max K=10; PCA per species |
| Stage 3: Feature construction | `src/build_features.py` | 8349-D vectors; **cosine similarity default** (euclidean fallback); diagnostic cluster-species table |
| MLP classifier + loss | `src/model.py` | `MotifClassifier` (BN+ReLU+Dropout MLP) + `MaskedFocalLoss` (per-sample/class masking) |
| Feature dataset | `src/dataset.py` | HDF5-backed, fold-split, pseudo-label injection support |
| Stage 4: Classifier training | `src/train_classifier.py` | AdamW + CosineAnnealingLR, wandb logging, fold via `FOLD` env var |
| Slurm scripts | `scripts/` | `extract_embeddings.sh`, `extract_distill_embeddings.sh`, `scrape_distill_data.sh`, `cluster.sh`, `train_classifier.sh`, `run_pipeline.sh` |
| Default config | `configs/default.yaml` | `similarity_metric: cosine`, `temperature: 0.1`, `top_k_logits: 200` |

### Outputs Produced ✅

| Output | Location | Status |
|--------|----------|--------|
| `segments.csv` | `outputs/segments.csv` | 358,454 rows |
| `embeddings.h5` | `outputs/embeddings/embeddings.h5` | 2.4 GB, 358,454 segments with spatial + logits |
| `global_prototypes.npz` | `outputs/prototypes/` | **(2048, 1536)** float32 — ready for prototype assignment |
| `global_hdbscan.pkl` | `outputs/prototypes/` | 237 MB — fitted HDBSCAN with `prediction_data=True` |
| `global_umap.pkl` | `outputs/prototypes/` | 4.1 GB — fitted UMAP transform |
| `global_cluster_stats.csv` | `outputs/prototypes/` | Per-cluster statistics |
| `species_gmms.pkl` | `outputs/prototypes/` | 234 species, PCA+GMM each |
| `distill_manifest.csv` | `data/distill_manifest.csv` | 3,806 rows (805 iNat + 3,001 BirdSet PER) |
| Distill audio | `data/distill_audio/` | 3,805 files |
| Pseudo-label files | `data/` | Multiple files exist: `pseudo_labels.csv`, `pseudo_labels_t06.csv`, `pseudo_labels_audio.csv`, `pseudo_labels_sc.csv` (and `_t06` variants) — likely from Perch logit bootstrap; **not yet integrated** into training pipeline |

### In Progress 🔄

| Job | Task | Status |
|-----|------|--------|
| Running (L40S) | Stage 0b + 1b: `extract_distill_embeddings.sh` (job 322207) | Stage 0b (`prepare_distill_data.py`) in progress at ~15% through 3,806 files; will then run Stage 1b (Perch embeddings on distill corpus) → `outputs/embeddings/distill_embeddings.h5` |
| Running (CPU) | Stage 3: `build_features.py` | Started 11:22, `features.h5` still growing (429 MB as of 15:22); no "Stage 3 complete" in log yet — expected to run several more hours |

### Newly Implemented ✅ (this session)

| Component | File(s) | Notes |
|-----------|---------|-------|
| Stage 4b: Pseudo-labeling | `src/pseudo_label.py` | Round 1: loads existing `data/pseudo_labels_*.csv` (Perch logit bootstrap, `filename,start,end,primary_label` schema); Round 2+: self-training MLP inference → threshold → `.npz` array |
| Stage 1d: Student model | `src/student_model.py` | `StudentEmbedder` (EfficientNet-B1-BirdSet + global/spatial heads) + `DistillationLoss` (cosine global + cosine spatial + MSE logits) |
| Stage 1d: Student training | `src/train_student.py` | Full training loop; uses `embeddings.h5` + `distill_embeddings.h5` as teacher targets; differential LR (backbone 1e-5, heads 1e-4); bfloat16 AMP; wandb |
| Stage 5: Inference | `src/inference.py` | `PantanalPredictor` class: student embed → prototype cosine assign → feature vector → MLP → per-segment probs; `predict_soundscape()` and `predict_directory()` for batch Kaggle inference; CLI entry point |
| Slurm: student training | `scripts/train_student.sh` | L40S, 12h, FOLD env var |
| Slurm: pseudo-labeling | `scripts/pseudo_label.sh` | L40S, 2h, ROUND + CHECKPOINT env vars |
| Config: stage1d + student_mel | `configs/default.yaml` | Added `stage1d` and `student_mel` sections |

### Still Missing ❌

| Component | Priority | Notes |
|-----------|----------|-------|
| **Distill features** | P2 | After job 322207 finishes (`distill_embeddings.h5`), run `build_features.py` on distill corpus. These aren't needed for student training (which reads raw audio + HDF5 targets directly), but would enable classifier training on expanded corpus. |
| **Temporal context model** | P4 | Sliding window GRU/1D-CNN over sequential windows. Defer until baseline validated. |
| **Cluster-species visualization notebook** | P5 | `cluster_species_table.npz` will be produced when `build_features.py` finishes; no notebook yet. |

### Architecture Note: Features vs. Student

`features.h5` was built using Perch (teacher) embeddings. The student will produce its own embeddings at inference. Because the student distillation loss is cosine similarity to Perch outputs, the student's embedding space aligns with Perch's — prototype assignment (just a matrix multiply against the same `global_prototypes.npz`) will work correctly with student embeddings. However, after student training, it is worth re-running `build_features.py` with student embeddings as a sanity check before finalizing MLP classifier weights.

---

## Future Improvements — Informed by Bird-MAE (arXiv:2504.12880)

**Paper**: "Can Masked Autoencoders Also Listen to Birds?" (Apr 2025). Introduces Bird-MAE, a ViT MAE pretrained on XCL (528k recordings, 9.7k species) with prototypical probing for frozen-representation classification. Achieves SOTA on BirdSet multi-label benchmarks, outperforming Perch by ~15pp mAP average across 8 tasks.

### Key findings relevant to our pipeline

1. **Learned prototypical probing crushes linear probing on frozen embeddings** — 49.97 mAP vs 13.29 mAP on HSN (37pp gap). Our pipeline uses unsupervised prototypes (HDBSCAN) + handcrafted features (histograms, spreads, entropies) + MLP. Their approach of learning J=20 class-specific prototypes with cosine max-pooling over spatial patches is simpler and stronger.

2. **Bird-MAE-L outperforms Perch on every benchmark** — range: +7pp to +16.4pp mAP across 8 BirdSet tasks. Our backbone is Perch v2. Switching to Bird-MAE would likely give a large lift, though at the cost of a heavier model (ViT-L = 304M params vs Perch's EfficientNet-B3 = 12M).

3. **Few-shot prototypical probing is remarkably strong** — 10-shot prototypical probing closely approaches full-data performance. This is directly relevant to our data-starved species problem.

### Concrete improvements to try (ordered by expected impact)

#### A. Replace HDBSCAN motifs + handcrafted features with learned prototypical probing
**What:** Instead of unsupervised HDBSCAN → motif histograms → MLP, learn J prototypes per class directly from labeled Perch embeddings. Classification becomes: cosine similarity between spatial embeddings and class prototypes → max-pool across spatial positions → constrained linear layer → sigmoid.

**Why:** Eliminates the weakest part of our pipeline — the handcrafted feature engineering between prototypes and classifier. Bird-MAE shows this is worth ~37pp mAP vs linear probing. Our HDBSCAN motifs are closer to this than linear probing, but still add unnecessary indirection.

**Key details from paper:**
- J=20 prototypes per class (total params: J×C×D + J×C + C ≈ 430k for 234 classes)
- Non-negative weight constraints on the linear layer (ensures prototypes contribute positively)
- Orthogonality loss during training (prevents prototype collapse)
- Cosine similarity, max-pooled over spatial dimensions
- Works on frozen embeddings — no backbone fine-tuning needed

**Compatibility:** Drops directly into our pipeline after Stage 1. Replaces Stages 2+3+4 with a single trainable module. Student distillation (Stage 1d) and pseudo-labeling (Stage 4b) remain unchanged.

#### B. Use Bird-MAE-Base as backbone instead of / alongside Perch
**What:** Replace Perch v2 embeddings with Bird-MAE-Base embeddings (ViT-B/16, 85M params, 768-D). Already available on HuggingFace: `DBD-research-group/Bird-MAE-Base`.

**Why:** 15pp average mAP improvement over Perch across all BirdSet tasks. Bird-MAE was pretrained on the same XCL data but with masked autoencoding rather than supervised classification, yielding richer representations.

**Tradeoffs:**
- Heavier than Perch — distillation even more critical for inference
- ViT architecture → different spatial embedding structure (8×64 patches vs Perch's 5×3)
- **Previously tested on BirdCallClassifier**: got reasonable results, but only Bird-MAE-Base (smallest variant) fits the Kaggle CPU inference budget. Larger variants (ViT-L, ViT-H) are too expensive without distillation. Same constraint as Perch — would need the same student distillation pipeline.
- If using Bird-MAE as teacher, the student distillation target changes from 1536-D (Perch) to 768-D (Bird-MAE-Base), which actually makes the student smaller and faster

#### C. Combine SupCon projection with prototypical probing
**What:** Use our existing SupCon projection (Stage 1.5) to pre-shape the embedding space, then learn prototypical probes on top. The contrastive projection pulls same-species embeddings together, making prototype learning easier.

**Why:** SupCon and prototypical probing address complementary aspects — SupCon optimizes the embedding geometry globally, prototypical probing optimizes class-specific decision boundaries locally.

#### D. Few-shot prototypical probing for data-starved species
**What:** For the ~14 species with ≤5 training samples, prototypical probing naturally handles low-resource classes (paper shows 10-shot approaching full-data performance). No special-case handling needed.

**Why:** Our current pipeline requires enough samples for GMM fitting (min_samples_per_component=20) — species below this threshold get no sub-cluster features. Prototypical probing has no such minimum.

### E. Student Distillation — Convergence Improvements (not yet implemented)

Ideas to push val_cos_sim beyond 0.92 (current best) toward 0.95+:

1. **EMA (Exponential Moving Average)**: Keep a shadow model with exponentially averaged weights (decay ~0.999). EMA models smooth out mini-batch noise and almost always outperform final weights for distillation. Essentially free — ~20 lines, no extra training time. Expected gain: +0.5-1% cos_sim.

2. **Hard example mining**: Once the model passes ~0.90, easy samples (loud, clear calls) contribute nearly zero gradient. Weight the loss by per-sample difficulty (1 - cos_sim) or drop samples where cos_sim > 0.98. Focuses GPU time on informative examples.

3. **Projection head trick** (from SimCLR/BYOL): Train with a small MLP projection head on top (student → 512 → 1536), but discard it at inference and use pre-projection features. The projection head absorbs training artifacts, leaving cleaner representations.

4. **Feature-level distillation**: Match intermediate backbone features in addition to final embeddings, giving the student richer gradient signal. Requires knowledge of Perch internals.

5. **Separate global/spatial cos_sim logging**: The blended metric understates global quality. Global alone is likely ~0.96 cos_sim already — logging it separately would clarify how good prototype assignment really is.

### Immediate Next Steps (ordered)

1. ~~Student distillation~~ — DONE (val_cos_sim=0.92, 60 epochs, batch_size=256, warmup+cosine LR)
2. **Extract student embeddings** — `sbatch scripts/extract_student_embeddings.sh`
3. **Rebuild features with student embeddings** — `python src/build_features.py --student`
4. **Retrain classifier on student features** — `sbatch scripts/train_classifier.sh`
5. **Package Kaggle submission** — bundle student ckpt + prototypes + classifier + inference.py
6. **Submit inference** — Kaggle notebook runs `PantanalPredictor` on test soundscapes
