"""
BirdCLEF 2026 — Kaggle Inference Notebook (Pantanal Pipeline, probing branch).
Student EfficientNet-B1 → spatial embeddings → prototypical probing head.

Constraints: CPU-only, no internet, 90-minute limit.
"""

import gc
import glob
import os
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

SAMPLE_RATE = CFG["data"]["sample_rate"]
SEGMENT_SECONDS = CFG["data"]["segment_duration"]
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_SECONDS)
MEL_CFG = CFG.get("student_mel", {})
BATCH_SIZE = 8


# ── Label map ───────────────────────────────────────────────────────────────
def build_label_map(taxonomy_path):
    df = pd.read_csv(taxonomy_path)
    labels = sorted(df["primary_label"].astype(str).unique())
    return {lbl: i for i, lbl in enumerate(labels)}, labels


label_map, sorted_labels = build_label_map(TAXONOMY_PATH)
NUM_CLASSES = len(label_map)
print(f"Classes: {NUM_CLASSES}")

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


# ── Prototypical Probing Head ───────────────────────────────────────────────
class PrototypicalHead(nn.Module):
    def __init__(self, embed_dim, num_classes, prototypes_per_class):
        super().__init__()
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, prototypes_per_class, embed_dim) * 0.02
        )
        self.raw_weights = nn.Parameter(
            torch.full((num_classes, prototypes_per_class), -2.0)
        )
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, spatial):
        z = F.normalize(spatial, dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum("bld,cjd->blcj", z, p)
        pooled = sim.max(dim=1).values
        w = F.softplus(self.raw_weights)
        return (pooled * w.unsqueeze(0)).sum(dim=-1) + self.bias


# ── Load components ─────────────────────────────────────────────────────────
t_load = time.time()

print("Loading student...")
student = StudentEmbedder.from_pretrained()
ckpt = torch.load(os.path.join(ARTIFACTS, "student_best.pt"),
                   map_location="cpu", weights_only=False)
student.load_state_dict(ckpt["model_state_dict"])
student.eval()
del ckpt; gc.collect()

print("Loading probe...")
probe_ckpt = torch.load(os.path.join(ARTIFACTS, "proto_probe_best.pt"),
                         map_location="cpu", weights_only=False)
pc = probe_ckpt["config"]
head = PrototypicalHead(embed_dim=int(pc["D"]),
                         num_classes=int(pc["num_classes"]),
                         prototypes_per_class=int(pc["J"]))
head.load_state_dict(probe_ckpt["model_state_dict"])
head.eval()
del probe_ckpt; gc.collect()
print(f"Loaded in {time.time() - t_load:.1f}s")


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
def predict_batch(waveforms):
    mels_np = np.stack([wav_to_mel_np(w, SAMPLE_RATE, MEL_CFG) for w in waveforms])
    mels_t = torch.from_numpy(mels_np).unsqueeze(1).float()
    _g, spatial = student(mels_t, normalize=True)
    logits = head(spatial)
    return torch.sigmoid(logits).numpy()


def main():
    t0 = time.time()

    test_files = sorted(
        glob.glob(os.path.join(TEST_SOUNDSCAPES, "*.ogg"))
        + glob.glob(os.path.join(TEST_SOUNDSCAPES, "*.wav"))
        + glob.glob(os.path.join(TEST_SOUNDSCAPES, "*.mp3"))
        + glob.glob(os.path.join(TEST_SOUNDSCAPES, "*.flac"))
    )
    print(f"Found {len(test_files)} test soundscapes")

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
        row_ids = [f"{fname_stem}_{(i + 1) * int(SEGMENT_SECONDS)}"
                   for i in range(n_segments)]

        for batch_start in range(0, n_segments, BATCH_SIZE):
            batch = segments[batch_start:batch_start + BATCH_SIZE]
            probs = predict_batch(batch)
            all_probs.append(probs[:, col_indices])
            all_row_ids.extend(row_ids[batch_start:batch_start + len(batch)])

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
