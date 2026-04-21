"""Stage 5: CPU inference pipeline for Kaggle submission.

Full pipeline:
  soundscape .ogg → 5s segments → mel spectrogram → Student embedder
  → cosine similarity to frozen prototypes → feature vector
  → MotifClassifier MLP → per-segment species logits
  → clip-level aggregation (max over overlapping windows)
  → submission CSV

Usage (Kaggle notebook):
    from inference import PantanalPredictor
    predictor = PantanalPredictor.load(
        student_ckpt="outputs/checkpoints/student/best.pt",
        prototypes_dir="outputs/prototypes",
        classifier_ckpt="outputs/checkpoints/classifier/best.pt",
        config_path="configs/default.yaml",
    )
    submission = predictor.predict_soundscape("path/to/soundscape.ogg")

Usage (batch CLI):
    python src/inference.py --soundscapes data/test_soundscapes/ --output submission.csv
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.special import softmax

from config import get_config
from student_model import StudentEmbedder
from model import MotifClassifier
from utils import setup_logging, load_audio_segment, build_label_map
from build_features import (
    _l2_normalize,
    compute_global_motif_features,
    compute_species_subcluster_features,
)
from nmf_per_class import project_segments_to_features


class PantanalPredictor:
    """Inference wrapper for the full Pantanal pipeline.

    All components are loaded once and kept in memory. Designed for
    batch processing of Kaggle test soundscapes on CPU.
    """

    def __init__(self,
                 student: StudentEmbedder,
                 prototypes: np.ndarray,
                 species_gmms: dict,
                 classifier: MotifClassifier,
                 label_map: dict,
                 cfg: dict,
                 device: torch.device,
                 project_fn=None,
                 nmf_W_all: np.ndarray | None = None,
                 nmf_W_all_pinv: np.ndarray | None = None,
                 nmf_boundaries: np.ndarray | None = None):
        self.student = student.to(device).eval()
        self.prototypes = prototypes          # (K, proj_dim) float32
        self.species_gmms = species_gmms
        self.classifier = classifier.to(device).eval()
        self.label_map = label_map
        self.inv_label_map = {v: k for k, v in label_map.items()}
        self.cfg = cfg
        self.device = device
        self.num_classes = len(label_map)
        self.project_fn = project_fn

        # Optional per-class NMF dictionaries
        self.nmf_W_all = nmf_W_all
        self.nmf_W_all_pinv = nmf_W_all_pinv
        self.nmf_boundaries = nmf_boundaries

        self.seg_dur = cfg["data"]["segment_duration"]
        self.sr = cfg["data"]["sample_rate"]
        self.metric = cfg["stage3"].get("similarity_metric", "cosine")
        self.temperature = cfg["stage3"].get("temperature", 0.1)
        self.mel_cfg = cfg.get("student_mel", {})

    @classmethod
    def load(cls, student_ckpt: str, prototypes_dir: str,
             classifier_ckpt: str, config_path: str = None,
             device: str = "cpu") -> "PantanalPredictor":
        """Load all components from disk."""
        from config import get_config
        cfg = get_config(config_path)
        dev = torch.device(device)

        # Student
        student_state = torch.load(student_ckpt, map_location=dev)
        student = StudentEmbedder.from_pretrained()
        student.load_state_dict(student_state["model_state_dict"])

        # Prototypes + GMMs
        proto_data = np.load(Path(prototypes_dir) / "global_prototypes.npz")
        prototypes = proto_data["prototypes"].astype(np.float32)

        gmm_path = Path(prototypes_dir) / "species_gmms.pkl"
        with open(gmm_path, "rb") as f:
            species_gmms = pickle.load(f)

        # SupCon projection (if prototypes were built in projected space)
        project_fn = None
        used_supcon = bool(proto_data.get("used_supcon", False))
        if used_supcon:
            W_path = Path(prototypes_dir) / "supcon_W.npy"
            if W_path.exists():
                W = np.load(W_path)  # (proj_dim, 1536)
                def project_fn(X):
                    orig_shape = X.shape[:-1]
                    flat = X.reshape(-1, X.shape[-1])
                    z = flat @ W.T
                    norms = np.linalg.norm(z, axis=1, keepdims=True)
                    z = z / np.maximum(norms, 1e-8)
                    return z.reshape(*orig_shape, W.shape[0])

        # Label map
        label_map = build_label_map(cfg["data"]["taxonomy_csv"])

        # Classifier
        ckpt = torch.load(classifier_ckpt, map_location=dev)
        feat_dim = ckpt["feat_dim"]
        num_classes = ckpt["num_classes"]
        classifier_cfg = ckpt.get("config", {})
        classifier = MotifClassifier(
            input_dim=feat_dim,
            num_classes=num_classes,
            hidden_dims=classifier_cfg.get("hidden_dims", [512, 256]),
            dropout=0.0,
        )
        classifier.load_state_dict(ckpt["model_state_dict"])

        # Optional per-class NMF dictionaries (stage_nmf_pc)
        nmf_W_all = nmf_W_all_pinv = nmf_boundaries = None
        nmf_dir_cfg = cfg["outputs"].get("nmf_dir", "")
        nmf_candidates = [Path(nmf_dir_cfg)] if nmf_dir_cfg else []
        # Also look next to the classifier checkpoint (packaged artifacts)
        nmf_candidates.append(Path(classifier_ckpt).parent)
        for nmf_dir in nmf_candidates:
            if (nmf_dir / "W_all_pinv.npy").exists():
                nmf_W_all_pinv = np.load(nmf_dir / "W_all_pinv.npy")
                nmf_boundaries = np.load(nmf_dir / "species_boundaries.npy")
                if (nmf_dir / "W_all.npy").exists():
                    nmf_W_all = np.load(nmf_dir / "W_all.npy")
                break

        return cls(student, prototypes, species_gmms, classifier,
                   label_map, cfg, dev, project_fn=project_fn,
                   nmf_W_all=nmf_W_all, nmf_W_all_pinv=nmf_W_all_pinv,
                   nmf_boundaries=nmf_boundaries)

    # ── Core embedding + feature computation ────────────────────────────────

    def _wav_to_mel(self, wav: np.ndarray) -> np.ndarray:
        """Waveform → mel spectrogram (n_mels, T) numpy."""
        from cache_mels import wav_to_mel_np
        return wav_to_mel_np(wav, self.sr, self.mel_cfg)  # (n_mels, T)

    @torch.no_grad()
    def _embed_batch(self, wavs: list[np.ndarray]
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Embed a batch of waveforms → (global_embs, spatial_embs, mels).

        Returns:
            global_embs: (B, 1536)
            spatial_embs: (B, 5, 1536)
            mels:        (B, n_mels, T) float32 — reused for NMF features
        """
        mels_np = np.stack([self._wav_to_mel(w) for w in wavs])  # (B, n_mels, T)
        mels_t = torch.from_numpy(mels_np).unsqueeze(1).float().to(self.device)
        g, s = self.student(mels_t, normalize=True)
        return g.cpu().numpy(), s.cpu().numpy(), mels_np

    def _build_feature(self, global_emb: np.ndarray,
                        spatial_emb: np.ndarray,
                        nmf_feat: np.ndarray | None = None) -> np.ndarray:
        """Build feature vector for one segment.

        If SupCon projection is available, project embeddings into the
        prototype space before computing motif/GMM features. The raw
        1536-D global embedding is still prepended to the feature vector
        (matching build_features.py behavior).

        nmf_feat: optional (2*num_species,) per-class NMF features appended
        after the subcluster features (matches build_features.py layout).
        """
        # Project into SupCon space for prototype comparison
        if self.project_fn is not None:
            spatial_proj = self.project_fn(spatial_emb)
            global_proj = self.project_fn(global_emb.reshape(1, -1))[0]
        else:
            spatial_proj = spatial_emb
            global_proj = global_emb

        hist, max_act, spread, noise = compute_global_motif_features(
            spatial_proj, self.prototypes,
            temperature=self.temperature,
            metric=self.metric,
        )
        best_match, sub_ent = compute_species_subcluster_features(
            global_proj, self.species_gmms, self.label_map,
        )

        parts = [global_emb, hist, max_act, spread, np.array([noise]),
                 best_match, sub_ent]
        if nmf_feat is not None:
            parts.append(nmf_feat)
        return np.concatenate(parts).astype(np.float32)

    def _compute_nmf_batch(self, mels: np.ndarray) -> np.ndarray | None:
        """Compute (B, 2*num_species) NMF features for a batch, or None."""
        if (self.nmf_W_all is None or self.nmf_W_all_pinv is None
                or self.nmf_boundaries is None):
            return None
        mels_nn = np.maximum(mels.astype(np.float32), 0.0)
        return project_segments_to_features(
            mels_nn, self.nmf_W_all, self.nmf_W_all_pinv, self.nmf_boundaries,
        )

    @torch.no_grad()
    def _classify_features(self, features: np.ndarray) -> np.ndarray:
        """Run MLP on (N, D) feature matrix → (N, C) probabilities."""
        feat_t = torch.from_numpy(features).to(self.device)
        batch_size = 512
        all_probs = []
        for start in range(0, len(feat_t), batch_size):
            logits = self.classifier(feat_t[start:start + batch_size])
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(all_probs, axis=0)

    # ── Soundscape-level prediction ──────────────────────────────────────────

    def predict_soundscape(self, audio_path: str,
                            stride_sec: float = 5.0,
                            batch_size: int = 16) -> pd.DataFrame:
        """Predict species for an entire soundscape file.

        Args:
            audio_path: path to .ogg/.wav soundscape
            stride_sec: stride between segments (5.0 = non-overlapping)
            batch_size: segments per forward pass

        Returns:
            DataFrame with columns: [row_id, *species_labels]
            where row_id = f"{filename}_{start_sec}"
        """
        import librosa

        filename = Path(audio_path).name
        wav_full, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        duration = len(wav_full) / self.sr

        # Build segment list
        starts = np.arange(0, duration - self.seg_dur + 1e-6, stride_sec)
        segments = []
        for s in starts:
            e = min(s + self.seg_dur, duration)
            chunk = wav_full[int(s * self.sr):int(e * self.sr)]
            segments.append((s, chunk))

        if not segments:
            return pd.DataFrame()

        # Process in batches
        all_probs = []
        all_row_ids = []

        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            starts_batch = [b[0] for b in batch]
            wavs_batch = [b[1] for b in batch]

            global_embs, spatial_embs, mels_batch = self._embed_batch(wavs_batch)
            nmf_batch = self._compute_nmf_batch(mels_batch)

            features = np.stack([
                self._build_feature(
                    global_embs[j], spatial_embs[j],
                    nmf_feat=None if nmf_batch is None else nmf_batch[j],
                )
                for j in range(len(wavs_batch))
            ])

            probs = self._classify_features(features)
            all_probs.append(probs)

            for s in starts_batch:
                end_sec = int(s + self.seg_dur)
                row_id = f"{Path(audio_path).stem}_{end_sec}"
                all_row_ids.append(row_id)

        all_probs = np.concatenate(all_probs, axis=0)

        # Build output DataFrame
        species_cols = [self.inv_label_map[i] for i in range(self.num_classes)]
        df = pd.DataFrame(all_probs, columns=species_cols)
        df.insert(0, "row_id", all_row_ids)
        return df

    def predict_directory(self, soundscape_dir: str,
                           stride_sec: float = 5.0,
                           batch_size: int = 16) -> pd.DataFrame:
        """Run predict_soundscape on all .ogg/.wav files in a directory."""
        import glob
        files = sorted(glob.glob(f"{soundscape_dir}/*.ogg")
                       + glob.glob(f"{soundscape_dir}/*.wav"))
        all_dfs = []
        for f in files:
            df = self.predict_soundscape(f, stride_sec=stride_sec,
                                          batch_size=batch_size)
            all_dfs.append(df)
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(cfg: dict):
    logger = setup_logging("inference", cfg["outputs"]["logs_dir"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--soundscapes", type=str,
                        default=str(Path(cfg["data"]["soundscape_dir"]).parent / "test_soundscapes"))
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--student_ckpt", type=str, default="")
    parser.add_argument("--classifier_ckpt", type=str, default="")
    parser.add_argument("--stride", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cpu")
    args, _ = parser.parse_known_args()

    # Auto-find best checkpoints if not specified
    ckpt_dir = Path(cfg["outputs"]["checkpoints_dir"])

    student_ckpt = args.student_ckpt
    if not student_ckpt:
        candidates = sorted((ckpt_dir / "student").glob("*.pt"),
                             key=lambda p: p.stat().st_mtime)
        if candidates:
            student_ckpt = str(candidates[-1])
            logger.info(f"Auto-selected student checkpoint: {student_ckpt}")
        else:
            raise FileNotFoundError("No student checkpoint found. Run train_student.py first.")

    classifier_ckpt = args.classifier_ckpt
    if not classifier_ckpt:
        candidates = sorted(ckpt_dir.rglob("best_val_auc*.pt"),
                             key=lambda p: p.stat().st_mtime)
        if candidates:
            classifier_ckpt = str(candidates[-1])
            logger.info(f"Auto-selected classifier checkpoint: {classifier_ckpt}")
        else:
            raise FileNotFoundError("No classifier checkpoint found. Run train_classifier.py first.")

    logger.info("Loading predictor...")
    predictor = PantanalPredictor.load(
        student_ckpt=student_ckpt,
        prototypes_dir=cfg["outputs"]["prototypes_dir"],
        classifier_ckpt=classifier_ckpt,
        device=args.device,
    )

    logger.info(f"Predicting soundscapes in {args.soundscapes} (stride={args.stride}s)...")
    submission = predictor.predict_directory(args.soundscapes,
                                              stride_sec=args.stride)

    submission.to_csv(args.output, index=False)
    logger.info(f"Submission saved → {args.output}  ({len(submission)} rows)")


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
