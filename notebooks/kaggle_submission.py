"""
BirdCLEF 2026 — Kaggle Inference Notebook (Pantanal Pipeline)
Student EfficientNet-B1 → SupCon-projected prototypes → MLP classifier.

Constraints: CPU-only, no internet, 90-minute limit.
"""

import gc
import glob
import os
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import librosa
import yaml
from scipy.special import softmax
from scipy.stats import entropy

# ── CPU optimization ────────────────────────────────────────────────────────
torch.set_num_threads(4)
torch.set_grad_enabled(False)

# ── Paths ───────────────────────────────────────────────────────────────────
_INPUT = "/kaggle/input"

COMPETITION_DATA = None
for p in [f"{_INPUT}/birdclef-2026", f"{_INPUT}/competitions/birdclef-2026"]:
    if os.path.isfile(os.path.join(p, "sample_submission.csv")):
        COMPETITION_DATA = p
        break
assert COMPETITION_DATA, f"Competition data not found under {_INPUT}"

ARTIFACTS = None
for p in [f"{_INPUT}/pantanal-artifacts",
          f"{_INPUT}/datasets/arimarkowitz/pantanal-artifacts"]:
    if os.path.isdir(p):
        ARTIFACTS = p
        break
assert ARTIFACTS, f"Artifacts dataset not found under {_INPUT}"

TAXONOMY_PATH = os.path.join(COMPETITION_DATA, "taxonomy.csv")
TEST_SOUNDSCAPES = os.path.join(COMPETITION_DATA, "test_soundscapes")
SAMPLE_SUB_PATH = os.path.join(COMPETITION_DATA, "sample_submission.csv")
OUTPUT_PATH = "/kaggle/working/submission.csv"

print(f"Competition data: {COMPETITION_DATA}")
print(f"Artifacts:        {ARTIFACTS}")

# ── Config ──────────────────────────────────────────────────────────────────
with open(os.path.join(ARTIFACTS, "config.yaml")) as f:
    CFG = yaml.safe_load(f)

SAMPLE_RATE = CFG["data"]["sample_rate"]          # 32000
SEGMENT_SECONDS = CFG["data"]["segment_duration"]  # 5.0
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_SECONDS)
MEL_CFG = CFG.get("student_mel", {})
TEMPERATURE = CFG["stage3"].get("temperature", 0.1)
METRIC = CFG["stage3"].get("similarity_metric", "cosine")
BATCH_SIZE = 8


# ── Label map ───────────────────────────────────────────────────────────────
def build_label_map(taxonomy_path):
    df = pd.read_csv(taxonomy_path)
    labels = sorted(df["primary_label"].astype(str).unique())
    return {lbl: i for i, lbl in enumerate(labels)}, labels


label_map, sorted_labels = build_label_map(TAXONOMY_PATH)
NUM_CLASSES = len(label_map)
print(f"Classes: {NUM_CLASSES}")

# Read submission column order and build index mapping
sample_sub = pd.read_csv(SAMPLE_SUB_PATH, nrows=0)
sub_columns = [c for c in sample_sub.columns if c != "row_id"]
assert len(sub_columns) == NUM_CLASSES, (
    f"Column mismatch: submission has {len(sub_columns)}, taxonomy has {NUM_CLASSES}"
)
col_indices = np.array([label_map[c] for c in sub_columns])


# ── Mel spectrogram ─────────────────────────────────────────────────────────
_mel_basis_cache = {}

def _build_mel_basis(sr, mel_cfg):
    n_mels = mel_cfg.get("n_mels", 128)
    hop_length = int(sr * mel_cfg.get("hop_ms", 10) / 1000)
    win_length = int(sr * mel_cfg.get("win_ms", 25) / 1000)
    n_fft = max(512, win_length)
    fmin = mel_cfg.get("fmin", 60.0)
    fmax = mel_cfg.get("fmax", 16000.0)
    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                  fmin=fmin, fmax=fmax)
    return {
        "mel_fb": mel_fb.astype(np.float32),
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
    }


def wav_to_mel_np(wav, sr, mel_cfg):
    """~3ms per 5s segment. Pure numpy, no torch dependency."""
    key = (sr, tuple(sorted(mel_cfg.items())))
    if key not in _mel_basis_cache:
        _mel_basis_cache[key] = _build_mel_basis(sr, mel_cfg)
    basis = _mel_basis_cache[key]

    if len(wav) < SEGMENT_SAMPLES:
        wav = np.pad(wav, (0, SEGMENT_SAMPLES - len(wav)))
    else:
        wav = wav[:SEGMENT_SAMPLES]

    S = np.abs(librosa.stft(wav, n_fft=basis["n_fft"],
                            hop_length=basis["hop_length"],
                            win_length=basis["win_length"])) ** 2
    mel = basis["mel_fb"] @ S
    mel_db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    mel_db = np.maximum(mel_db, mel_db.max() - 80.0)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype(np.float32)


# ── Student Embedder ────────────────────────────────────────────────────────
from transformers import EfficientNetModel

PERCH_EMBED_DIM = 1536
SPATIAL_H = 5

BIRDSET_MODEL_DIR = os.path.join(ARTIFACTS, "efficientnet_b1_birdset")
assert os.path.isfile(os.path.join(BIRDSET_MODEL_DIR, "config.json")), (
    f"BirdSet config not found at {BIRDSET_MODEL_DIR}. "
    "Add efficientnet_b1_birdset/ to the Kaggle dataset."
)


class StudentEmbedder(nn.Module):
    def __init__(self, backbone, backbone_hidden=1280,
                 embed_dim=PERCH_EMBED_DIM, spatial_h=SPATIAL_H):
        super().__init__()
        self.backbone = backbone
        self.spatial_h = spatial_h
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_hidden, embed_dim),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((spatial_h, 1))
        self.spatial_proj = nn.Linear(backbone_hidden, embed_dim)

    @classmethod
    def from_pretrained(cls, backbone_path=BIRDSET_MODEL_DIR,
                        embed_dim=PERCH_EMBED_DIM):
        backbone = EfficientNetModel.from_pretrained(backbone_path)
        hidden = backbone.config.hidden_dim
        return cls(backbone, backbone_hidden=hidden, embed_dim=embed_dim)

    def forward(self, mel, normalize=True):
        feat = self.backbone(mel).last_hidden_state
        global_emb = self.global_head(feat)
        spatial_pooled = self.spatial_pool(feat)
        B, C, H, W = spatial_pooled.shape
        cells = spatial_pooled.squeeze(-1).permute(0, 2, 1).reshape(-1, C)
        cells_proj = self.spatial_proj(cells)
        spatial_emb = cells_proj.reshape(B, H, -1)
        if normalize:
            global_emb = F.normalize(global_emb, dim=-1)
            spatial_emb = F.normalize(spatial_emb, dim=-1)
        return global_emb, spatial_emb


# ── MLP Classifier ──────────────────────────────────────────────────────────
class MotifClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=(512, 256), dropout=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


# ── Feature computation ─────────────────────────────────────────────────────
def _l2_normalize(x, axis=-1, eps=1e-8):
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norms, eps)


def compute_global_motif_features(spatial_emb, prototypes,
                                   temperature=0.1, metric="cosine"):
    local = spatial_emb.reshape(-1, spatial_emb.shape[-1])
    K = prototypes.shape[0]
    if metric == "cosine":
        local_n = _l2_normalize(local, axis=1)
        proto_n = _l2_normalize(prototypes, axis=1)
        logits = (local_n @ proto_n.T) / temperature
    else:
        z_sq = np.sum(local ** 2, axis=1, keepdims=True)
        p_sq = np.sum(prototypes ** 2, axis=1, keepdims=True).T
        logits = -(z_sq + p_sq - 2.0 * local @ prototypes.T) / temperature

    weights = softmax(logits, axis=1)
    hist = weights.sum(axis=0)
    max_act = weights.max(axis=0)
    spread = weights.std(axis=0)
    avg_ent = np.mean([entropy(w) for w in weights])
    max_ent = np.log(K) if K > 1 else 1.0
    noise = avg_ent / max_ent
    return hist, max_act, spread, noise


def compute_species_subcluster_features(global_emb, species_gmms, label_map):
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
            reduced = model["pca"].transform(emb_2d)
            posteriors = model["gmm"].predict_proba(reduced)[0]
            best_match[idx] = posteriors.max()
            sub_entropy[idx] = entropy(posteriors + 1e-10)
        except Exception:
            continue
    return best_match, sub_entropy


# ── Load all components ─────────────────────────────────────────────────────
t_load = time.time()

print("Loading student model...")
student = StudentEmbedder.from_pretrained()
ckpt = torch.load(os.path.join(ARTIFACTS, "student_best.pt"),
                   map_location="cpu", weights_only=False)
student.load_state_dict(ckpt["model_state_dict"])
student.eval()
del ckpt; gc.collect()
print(f"  Student loaded")

print("Loading prototypes and GMMs...")
proto_data = np.load(os.path.join(ARTIFACTS, "global_prototypes.npz"))
prototypes = proto_data["prototypes"].astype(np.float32)
W = np.load(os.path.join(ARTIFACTS, "supcon_W.npy"))  # (proj_dim, 1536)
print(f"  Prototypes: {prototypes.shape}, SupCon projection: 1536 -> {W.shape[0]}")


def project_fn(X):
    """Project (..., 1536) -> (..., proj_dim) with L2 normalization."""
    orig_shape = X.shape[:-1]
    flat = X.reshape(-1, X.shape[-1])
    z = flat @ W.T
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
    return z.reshape(*orig_shape, W.shape[0])


with open(os.path.join(ARTIFACTS, "species_gmms.pkl"), "rb") as f:
    species_gmms = pickle.load(f)

print("Loading classifier...")
cls_ckpt = torch.load(os.path.join(ARTIFACTS, "classifier_best.pt"),
                       map_location="cpu", weights_only=False)
feat_dim = cls_ckpt["feat_dim"]
cls_cfg = cls_ckpt.get("config", {})
classifier = MotifClassifier(
    input_dim=feat_dim,
    num_classes=cls_ckpt["num_classes"],
    hidden_dims=cls_cfg.get("hidden_dims", [512, 256]),
    dropout=0.0,
)
classifier.load_state_dict(cls_ckpt["model_state_dict"])
classifier.eval()
del cls_ckpt; gc.collect()
print(f"  Classifier: {feat_dim}D -> {NUM_CLASSES} classes")
print(f"All models loaded in {time.time() - t_load:.1f}s")


# ── Audio helpers ───────────────────────────────────────────────────────────
def load_audio(path):
    try:
        audio, sr = sf.read(path, dtype="float32")
    except Exception:
        audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        return audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32)


def segment_audio(audio):
    n_segments = len(audio) // SEGMENT_SAMPLES
    if n_segments == 0:
        return None
    segments = np.empty((n_segments, SEGMENT_SAMPLES), dtype=np.float32)
    for i in range(n_segments):
        segments[i] = audio[i * SEGMENT_SAMPLES:(i + 1) * SEGMENT_SAMPLES]
    return segments


# ── Inference ───────────────────────────────────────────────────────────────
def embed_batch(waveforms):
    """Batch of waveforms (B, samples) -> global (B, 1536), spatial (B, 5, 1536), mels (B, n_mels, T)."""
    mels_np = np.stack([wav_to_mel_np(w, SAMPLE_RATE, MEL_CFG) for w in waveforms])  # (B, n_mels, T)
    mels_t = torch.from_numpy(mels_np).unsqueeze(1).float()  # (B, 1, n_mels, T)
    g, s = student(mels_t, normalize=True)
    return g.numpy(), s.numpy(), mels_np


def build_features_batch(global_embs, spatial_embs, mels_batch=None):
    """Build feature vectors for a batch of segments."""
    features = []
    for j in range(len(global_embs)):
        g = global_embs[j]
        s = spatial_embs[j]

        s_proj = project_fn(s)
        g_proj = project_fn(g.reshape(1, -1))[0]

        hist, max_act, spread, noise = compute_global_motif_features(
            s_proj, prototypes, temperature=TEMPERATURE, metric=METRIC)
        best_match, sub_ent = compute_species_subcluster_features(
            g_proj, species_gmms, label_map)

        parts = [g, hist, max_act, spread, [noise], best_match, sub_ent]
        feat = np.concatenate(parts).astype(np.float32)
        features.append(feat)
    return np.stack(features)


def classify_features(features):
    """(N, D) feature matrix -> (N, C) probabilities."""
    feat_t = torch.from_numpy(features)
    all_probs = []
    for start in range(0, len(feat_t), 512):
        logits = classifier(feat_t[start:start + 512])
        all_probs.append(torch.sigmoid(logits).numpy())
    return np.concatenate(all_probs, axis=0)


# ── Main loop ──────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    test_files = sorted(
        glob.glob(os.path.join(TEST_SOUNDSCAPES, "*.ogg"))
        + glob.glob(os.path.join(TEST_SOUNDSCAPES, "*.wav"))
        + glob.glob(os.path.join(TEST_SOUNDSCAPES, "*.mp3"))
        + glob.glob(os.path.join(TEST_SOUNDSCAPES, "*.flac"))
    )
    print(f"\nFound {len(test_files)} test soundscapes")

    all_row_ids = []
    all_probs = []

    for file_idx, filepath in enumerate(test_files):
        fname_stem = os.path.splitext(os.path.basename(filepath))[0]

        try:
            audio = load_audio(filepath)
        except Exception as e:
            print(f"WARNING: Failed to load {filepath}: {e}")
            continue

        segments = segment_audio(audio)
        if segments is None:
            continue

        n_segments = len(segments)
        # row_id = filename_without_ext + "_" + end_second
        row_ids = [f"{fname_stem}_{(i + 1) * int(SEGMENT_SECONDS)}"
                   for i in range(n_segments)]

        for batch_start in range(0, n_segments, BATCH_SIZE):
            batch = segments[batch_start:batch_start + BATCH_SIZE]

            global_embs, spatial_embs, mels_batch = embed_batch(batch)
            features = build_features_batch(global_embs, spatial_embs, mels_batch)
            probs = classify_features(features)

            # Reorder columns to match submission format
            all_probs.append(probs[:, col_indices])
            all_row_ids.extend(
                row_ids[batch_start:batch_start + len(batch)])

        if (file_idx + 1) % 50 == 0:
            print(f"  Processed {file_idx + 1}/{len(test_files)} "
                  f"({time.time() - t0:.0f}s)")

    print(f"Building submission with {len(all_row_ids)} rows...")
    if not all_probs:
        sub = pd.read_csv(SAMPLE_SUB_PATH)
        sub.to_csv(OUTPUT_PATH, index=False)
        print(f"No test audio found — wrote sample submission to {OUTPUT_PATH}")
        return

    probs_matrix = np.concatenate(all_probs, axis=0)
    sub = pd.DataFrame(probs_matrix, columns=sub_columns)
    sub.insert(0, "row_id", all_row_ids)
    sub.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH} ({len(sub)} rows, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
