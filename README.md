# PantanalSpeciesMonitoring

## Architecture (Concise)

### Goal

Build a **multi-taxa bioacoustic model (birds, frogs, insects)** that:

* performs strong classification
* learns **recurring acoustic motifs**
* remains **deployable (CPU-friendly)**

---

## 1. Pipeline Overview

```text
audio → spectrogram → encoder → embeddings
      → clustering (global + class)
      → motif features
      → classifier
```

---

## 2. Components

### A. Preprocessing

* mel / log-mel / PCEN spectrograms
* standard augmentations (time shift, mixup, etc.)

---

### B. Encoder (flexible)

Options:

* EfficientNet (baseline)
* HTS-AT
* foundation model (e.g. Perch, if feasible)

Outputs:

* **global embedding (clip)**
* **local embeddings (frames/patches)** ← critical

---

### C. Motif Discovery (core idea)

#### Global clustering

* cluster all embeddings (HDBSCAN)
* learn shared acoustic “atoms” (insects, pulses, noise, etc.)

#### Class-specific clustering

* cluster embeddings within each class
* learn **what that species can sound like**

#### Harmonization

* represent each clip using:

  * global cluster activations
  * class-specific cluster activations

---

### D. Features

Per clip:

* histogram of cluster activations
* distances to cluster centers
* motif frequency / density

---

### E. Classifier

```text
encoder features + motif features → MLP → multi-label output
```

---

## 3. Training Strategy

1. Train baseline encoder + classifier
2. Extract local embeddings
3. Run:

   * global clustering
   * class-specific clustering
4. Build motif features
5. Train combined model

---

## 4. Inference Modes

* **Simple:** encoder + classifier
* **Structured:** encoder + cluster lookup + classifier
* **Optimized:** distilled lightweight model (for CPU)

---

## 5. Key Idea

Not:

> one sound per species

But:

> each species = **set of recurring acoustic motifs**

---

## 6. Notes

* Prefer **embedding clustering over NMFk** (cheaper, more robust)
* Keep clustering **offline**
* Keep inference **lightweight**

---

## 7. Milestones

1. Strong baseline (EfficientNet/HTS-AT)
2. Extract embeddings + cluster
3. Add motif features
4. Compare performance

---

## Summary

A hybrid system:

* **encoder learns features**
* **clustering discovers motifs**
* **classifier learns which motifs matter**

This balances performance, interpretability, and practicality.
