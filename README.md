# PantanalSpeciesMonitoring

## Architecture (Concise)

### Goal

Build a **multi-taxa bioacoustic model (birds, frogs, insects)** that:

* performs strong classification
* learns **recurring acoustic motifs**
* can point to **specific spectrogram patterns** that correspond to animals
* supports **overlapping signals via soft motif mixtures**
* remains **deployable (CPU-friendly)**

---

## 1. Pipeline Overview

```text
audio → multi-view spectrogram → local encoder → embeddings (Z)
      → global clustering → prototypes (P)
      → soft assignment (W)
      → motif features
      → classifier
```

---

## 2. Components

### A. Preprocessing (Multi-View Input)

* original log-mel
* PCEN (background suppression)
* frequency-boosted / suppressed variants

Stack as channels: `(freq × time × channels)`

Purpose: expose the **same signal under different conditions**.

---

### B. Local Encoder (Core)

Fully convolutional encoder that accepts **variable-length input** and outputs a grid of **local embeddings**:

```text
Z = {z_1, z_2, ..., z_T}
```

Each `z_t` represents a **local time–frequency receptive field**.

Optional: use a foundation model (e.g., Perch) as **teacher / prior**.

---

### C. Global Motif Discovery (Clustering)

Pool all local embeddings across the dataset and cluster:

```text
Z_all → clustering → P = {p_1, ..., p_K}
```

Each prototype `p_k` ≈ a **recurring acoustic motif** (frog pulse, insect texture, harmonic stack, noise, etc.).

Recommended: HDBSCAN (density-based, handles noise) or k-means (simpler baseline).

---

### D. Soft Assignment (Key Update)

Do **not** force one cluster per embedding.

For each local embedding `z_t`, compute **soft weights** over prototypes:

```text
w_t[k] = softmax(sim(z_t, p_k))
```

Interpretation:

* allows **overlapping signals** (e.g., frog + insect)
* `z_t` can be a **mixture of motifs**

This is a neural analogue of **mixture decomposition** (NMF-like, but learned in embedding space).

---

### E. Motif Features (Per File)

Aggregate soft assignments over time:

* motif histogram: `h_k = Σ_t w_t[k]`
* temporal traces: `w_t[k]` over time (for localization)
* persistence / density per motif

This yields a **bag-of-motifs** representation plus **time-local activations**.

---

### F. Classifier

```text
[encoder pooled features + motif features] → MLP → multi-label output
```

---

## 3. Training Strategy

1. Initialize encoder (pretrained if possible)
2. (Optional) multi-view **consistency** on local patches
3. Extract local embeddings `Z`
4. Cluster `Z_all` → prototypes `P`
5. Compute soft assignments `W`
6. Build motif features
7. Train classifier (weak labels)

Notes:

* clustering is **offline**
* labels are used to map motifs → species (not to define motifs)

---

## 4. Inference Modes

* **Simple:** encoder + classifier
* **Structured:** encoder → soft assignments → motif features → classifier
* **Explainable:** visualize `w_t[k]` over spectrogram ("this region is frog-like")

---

## 5. Key Ideas

* **Local embeddings** capture structure in small spectrogram regions
* **Clusters = motifs** discovered globally across data
* **Soft assignment** allows **multiple signals per region**
* **Sequences of activations** recover variable-length events

---

## 6. Notes

* Prefer **embedding clustering over NMFk** (scalable, robust)
* Use **soft clustering** to handle mixtures
* Keep encoder **general (pretrained) + lightly adapted**
* Multi-view inputs help enforce **signal invariance**

---

## 7. Milestones

1. Baseline encoder + multi-view input
2. Extract local embeddings
3. Global clustering → prototypes
4. Add soft assignment + motif features
5. Evaluate clustering quality + interpretability

---

## Summary

A hybrid system where:

* encoder learns **local acoustic structure**
* clustering discovers **global motifs**
* soft assignment represents **mixtures of signals**
* classifier maps motifs → species

Goal:

> identify and explain **which spectrogram patterns correspond to animals**, even under overlap
