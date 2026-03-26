# PantanalSpeciesMonitoring

## Architecture (Concise)

### Goal

Build a **multi-taxa bioacoustic model (birds, frogs, insects)** that:

* performs strong classification
* learns **recurring acoustic motifs**
* can isolate **specific spectrogram patterns** that correspond to animals
* remains **deployable (CPU-friendly)**

---

## 1. Pipeline Overview

```text
audio → multi-view spectrogram → local encoder → embeddings
      → clustering (global + class)
      → motif features
      → classifier
```

---

## 2. Components

### A. Preprocessing (Multi-View Input)

Instead of a single spectrogram, generate **multiple transformed views** of the same audio:

* original log-mel spectrogram
* PCEN (background suppression)
* frequency-boosted / suppressed variant

Stack as channels:

```
(freq × time × channels)
```

Purpose:

* expose the **same signal under different conditions**
* help distinguish **animal signal vs background**

---

### B. Local Encoder (Core)

Use a **fully convolutional spectrogram encoder** that:

* accepts **variable-length input**
* outputs **local embeddings** (not just clip-level)

Output:

```
(time_steps × embedding_dim)
```

Each embedding represents a **local acoustic pattern**.

Optional:

* use Perch-style embeddings as **semantic teacher features** (offline or auxiliary)

---

### C. Motif Discovery (Core Idea)

Clustering operates on **local embeddings**, not full clips.

#### Global clustering

* cluster all local embeddings (HDBSCAN)
* learn shared acoustic “atoms”:

  * insect textures
  * frog pulses
  * harmonics
  * noise types

#### Class-specific clustering

* cluster embeddings from positive clips per class
* learn **what that species can sound like**

#### Harmonization

Each local embedding is mapped to:

* a **global cluster** (shared motif)
* a **class-specific cluster** (species-specific motif)

---

### D. Motif Features

Aggregate over a clip:

* histogram of global cluster activations
* histogram of class-specific activations
* distances to cluster centers
* motif frequency / persistence

These represent:

> which acoustic patterns appear, and how strongly

---

### E. Classifier

```
encoder features + motif features → MLP → multi-label output
```

---

## 3. Training Strategy

1. Train baseline encoder + classifier
2. Extract **local embeddings**
3. Run:

   * global clustering
   * class-specific clustering
4. Score clusters using labels (keep discriminative motifs)
5. Build motif features
6. Train combined model

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

> each species = **set of recurring spectrogram patterns (motifs)**

Goal:

> identify **which specific patterns in the spectrogram correspond to animals**

---

## 6. Notes

* Prefer **embedding clustering over NMFk** (cheaper, more robust)
* Use **local embeddings**, not clip embeddings, for motif discovery
* Multi-view spectrograms help expose latent signals
* Keep clustering **offline**
* Keep inference **lightweight**

---

## 7. Milestones

1. Strong baseline (EfficientNet / CNN encoder)
2. Multi-view spectrogram input
3. Extract local embeddings + cluster
4. Add motif features
5. Evaluate interpretability + performance

---

## Summary

A hybrid system:

* **encoder learns local acoustic structure**
* **multi-view input exposes hidden signals**
* **clustering discovers recurring motifs**
* **classifier learns which motifs correspond to species**

This directly supports:

> identifying the **specific spectrogram patterns that are the animal**
