"""Stage 5: CPU inference with prototypical probing head.

Pipeline:
  soundscape .ogg → 5s segments → mel spectrogram → Student embedder
  → spatial embeddings (L, D) → PrototypicalHead → per-species logits
  → clip-level aggregation → submission CSV

Usage (Kaggle notebook):
    from inference import PantanalPredictor
    predictor = PantanalPredictor.load(
        student_ckpt="outputs/checkpoints/student/best.pt",
        probe_ckpt="outputs/checkpoints/proto_probe/fold0_best.pt",
        config_path="configs/default.yaml",
    )
    submission = predictor.predict_soundscape("path/to/soundscape.ogg")

Usage (batch CLI):
    python src/inference.py --soundscapes data/test_soundscapes/ --output submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import get_config
from student_model import StudentEmbedder
from proto_probe import PrototypicalHead
from utils import setup_logging, load_audio_segment, build_label_map


class PantanalPredictor:
    """Inference wrapper: Student encoder + PrototypicalHead."""

    def __init__(self,
                 student: StudentEmbedder,
                 head: PrototypicalHead,
                 label_map: dict,
                 cfg: dict,
                 device: torch.device):
        self.student = student.to(device).eval()
        self.head = head.to(device).eval()
        self.label_map = label_map
        self.inv_label_map = {v: k for k, v in label_map.items()}
        self.cfg = cfg
        self.device = device
        self.num_classes = len(label_map)

        self.seg_dur = cfg["data"]["segment_duration"]
        self.sr = cfg["data"]["sample_rate"]
        self.mel_cfg = cfg.get("student_mel", {})

    @classmethod
    def load(cls, student_ckpt: str, probe_ckpt: str,
             config_path: str | None = None, device: str = "cpu") -> "PantanalPredictor":
        cfg = get_config(config_path)
        dev = torch.device(device)

        student_state = torch.load(student_ckpt, map_location=dev)
        student = StudentEmbedder.from_pretrained()
        student.load_state_dict(student_state["model_state_dict"])

        probe_state = torch.load(probe_ckpt, map_location=dev)
        pc = probe_state["config"]
        head = PrototypicalHead(
            embed_dim=int(pc["D"]),
            num_classes=int(pc["num_classes"]),
            prototypes_per_class=int(pc["J"]),
        )
        head.load_state_dict(probe_state["model_state_dict"])

        label_map = build_label_map(cfg["data"]["taxonomy_csv"])
        return cls(student, head, label_map, cfg, dev)

    def _wav_to_mel(self, wav: np.ndarray) -> np.ndarray:
        from cache_mels import wav_to_mel_np
        return wav_to_mel_np(wav, self.sr, self.mel_cfg)

    @torch.no_grad()
    def _embed_batch(self, wavs: list[np.ndarray]) -> torch.Tensor:
        """Embed a batch of waveforms → spatial embeddings (B, L, D)."""
        mels_np = np.stack([self._wav_to_mel(w) for w in wavs])   # (B, n_mels, T)
        mels_t = torch.from_numpy(mels_np).unsqueeze(1).float().to(self.device)
        _g, spatial = self.student(mels_t, normalize=True)         # (B, L, D)
        return spatial

    @torch.no_grad()
    def _classify(self, spatial: torch.Tensor) -> np.ndarray:
        logits = self.head(spatial)
        return torch.sigmoid(logits).cpu().numpy()

    def predict_soundscape(self, audio_path: str,
                            stride_sec: float | None = None,
                            batch_size: int = 16) -> pd.DataFrame:
        """Slide a window across a soundscape and return per-window species probs."""
        stride = stride_sec if stride_sec is not None else self.seg_dur
        wav_full = load_audio_segment(audio_path, 0.0, None, sr=self.sr)
        duration = len(wav_full) / self.sr
        starts = np.arange(0.0, max(0.0, duration - self.seg_dur + 1e-6), stride)

        segments: list[tuple[float, np.ndarray]] = []
        for s in starts:
            a, b = int(s * self.sr), int((s + self.seg_dur) * self.sr)
            chunk = wav_full[a:b]
            if len(chunk) < int(self.seg_dur * self.sr):
                chunk = np.pad(chunk, (0, int(self.seg_dur * self.sr) - len(chunk)))
            segments.append((s, chunk))

        if not segments:
            return pd.DataFrame()

        all_probs, all_row_ids = [], []
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            starts_batch = [b[0] for b in batch]
            wavs_batch = [b[1] for b in batch]

            spatial = self._embed_batch(wavs_batch)
            probs = self._classify(spatial)
            all_probs.append(probs)

            for s in starts_batch:
                row_id = f"{Path(audio_path).stem}_{int(s + self.seg_dur)}"
                all_row_ids.append(row_id)

        all_probs = np.concatenate(all_probs, axis=0)
        species_cols = [self.inv_label_map[i] for i in range(self.num_classes)]
        df = pd.DataFrame(all_probs, columns=species_cols)
        df.insert(0, "row_id", all_row_ids)
        return df

    def predict_directory(self, soundscape_dir: str,
                           stride_sec: float | None = None,
                           batch_size: int = 16) -> pd.DataFrame:
        import glob
        files = sorted(glob.glob(f"{soundscape_dir}/*.ogg")
                       + glob.glob(f"{soundscape_dir}/*.wav"))
        all_dfs = []
        for f in files:
            df = self.predict_soundscape(f, stride_sec=stride_sec,
                                          batch_size=batch_size)
            all_dfs.append(df)
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--soundscapes", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--student-ckpt", type=str,
                        default="outputs/checkpoints/student/best.pt")
    parser.add_argument("--probe-ckpt", type=str,
                        default="outputs/checkpoints/proto_probe/fold0_best.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stride-sec", type=float, default=None)
    args = parser.parse_args()

    predictor = PantanalPredictor.load(
        student_ckpt=args.student_ckpt,
        probe_ckpt=args.probe_ckpt,
    )
    df = predictor.predict_directory(
        args.soundscapes,
        stride_sec=args.stride_sec,
        batch_size=args.batch_size,
    )
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows → {args.output}")


if __name__ == "__main__":
    main()
