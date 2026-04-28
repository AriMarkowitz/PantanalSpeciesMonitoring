"""Audio + mel augmentations for student distillation.

Only the STUDENT input is augmented. Teacher embeddings are frozen targets
produced from clean clips — we don't re-embed. That's the standard
distillation-with-augmentation pattern: force the student to reach the same
representation through harder-to-read inputs, which improves generalisation
without any teacher recomputation.

For mixup we take a convex combo of teacher embeddings on the fly; L2
re-normalisation absorbs the magnitude change.

Augmentations defined here:
  - WaveformAug:   time-shift, gain jitter, optional noise overlay
  - SpecAugment:   time & frequency masking on mel (Park et al., 2019)
  - mixup_pair:    batch-level mixup — returns mixed mel + convex-combined
                   teacher targets (global, spatial, logits)

Each is a plain callable; the training loop chains them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class WaveformAug:
    """Waveform-level augmentation applied BEFORE mel extraction.

    - Time shift up to ±shift_sec with circular wrap.
    - Random gain in [-gain_db, +gain_db] dB.
    - Background-noise mixing (if a noise pool is provided) at a random
      SNR in [snr_min, snr_max] dB with probability noise_p.
    """

    def __init__(self,
                 sr: int,
                 shift_sec: float = 0.5,
                 gain_db: float = 6.0,
                 noise_p: float = 0.5,
                 snr_min: float = 0.0,
                 snr_max: float = 20.0,
                 noise_pool: "NoisePool | None" = None,
                 rng: np.random.Generator | None = None):
        self.sr = sr
        self.shift_samples = int(shift_sec * sr)
        self.gain_db = gain_db
        self.noise_p = noise_p
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.noise_pool = noise_pool
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        # Time shift (cheap; np.roll is contiguous under the hood)
        if self.shift_samples > 0:
            shift = int(self.rng.integers(-self.shift_samples, self.shift_samples + 1))
            if shift != 0:
                wav = np.roll(wav, shift)

        # Random gain
        if self.gain_db > 0:
            db = float(self.rng.uniform(-self.gain_db, self.gain_db))
            wav = wav * (10.0 ** (db / 20.0))

        # Background noise overlay
        if self.noise_pool is not None and self.rng.random() < self.noise_p:
            noise = self.noise_pool.sample(len(wav), self.rng)
            if noise is not None:
                snr_db = float(self.rng.uniform(self.snr_min, self.snr_max))
                wav = _mix_snr(wav, noise, snr_db)

        return wav.astype(np.float32, copy=False)


def _mix_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix noise into signal at a target SNR (dB). Preserves signal scale."""
    sig_pow = float(np.mean(signal ** 2)) + 1e-10
    noise_pow = float(np.mean(noise ** 2)) + 1e-10
    target_noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
    scale = float(np.sqrt(target_noise_pow / noise_pow))
    return signal + scale * noise


class NoisePool:
    """Sliceable pool of background audio drawn from unlabeled soundscape clips.

    Loads a small random sample of full-duration audio files and serves
    `segment_samples`-length chunks at random offsets. Fork-safe (no HDF5
    handles). Files are re-loaded lazily on first access per worker.
    """

    def __init__(self, audio_paths: list[str], sr: int, segment_samples: int,
                 max_files: int = 100, rng: np.random.Generator | None = None):
        self.audio_paths = audio_paths
        self.sr = sr
        self.segment_samples = segment_samples
        self.max_files = max_files
        self.rng = rng if rng is not None else np.random.default_rng()
        self._buffers: list[np.ndarray] = []  # filled lazily per worker

    def _ensure_loaded(self, rng: np.random.Generator):
        if self._buffers or not self.audio_paths:
            return
        # Each worker picks its own random subset — cheap diversification
        picks = rng.choice(len(self.audio_paths),
                           size=min(self.max_files, len(self.audio_paths)),
                           replace=False)
        import soundfile as sf
        for i in picks:
            try:
                wav, file_sr = sf.read(self.audio_paths[int(i)], dtype="float32")
            except Exception:
                continue
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if file_sr != self.sr:
                import librosa
                wav = librosa.resample(wav, orig_sr=file_sr, target_sr=self.sr)
            if len(wav) >= self.segment_samples:
                self._buffers.append(wav.astype(np.float32))

    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray | None:
        self._ensure_loaded(rng)
        if not self._buffers:
            return None
        buf = self._buffers[int(rng.integers(len(self._buffers)))]
        if len(buf) <= n_samples:
            return buf[:n_samples]
        start = int(rng.integers(0, len(buf) - n_samples + 1))
        return buf[start:start + n_samples]


def spec_augment(mel: torch.Tensor,
                  time_mask_pct: float = 0.2,
                  freq_mask_pct: float = 0.2,
                  num_time_masks: int = 2,
                  num_freq_masks: int = 2) -> torch.Tensor:
    """SpecAugment on a single mel (shape (..., n_mels, T)).

    In-place zero-masking of up to `time_mask_pct` of frames and
    `freq_mask_pct` of mel bins. Paper: Park et al., 2019.
    Applied once per sample per epoch — the masks differ across epochs.
    """
    if mel.ndim == 2:
        n_mels, T = mel.shape
    else:
        n_mels, T = mel.shape[-2], mel.shape[-1]

    for _ in range(num_time_masks):
        t = int(torch.randint(0, max(1, int(T * time_mask_pct) + 1), (1,)).item())
        if t > 0 and T - t > 0:
            t0 = int(torch.randint(0, T - t, (1,)).item())
            mel[..., :, t0:t0 + t] = 0.0

    for _ in range(num_freq_masks):
        f = int(torch.randint(0, max(1, int(n_mels * freq_mask_pct) + 1), (1,)).item())
        if f > 0 and n_mels - f > 0:
            f0 = int(torch.randint(0, n_mels - f, (1,)).item())
            mel[..., f0:f0 + f, :] = 0.0
    return mel


def mixup_pair(mels: torch.Tensor,
               t_global: torch.Tensor,
               t_spatial: torch.Tensor,
               t_logits: torch.Tensor,
               alpha: float) -> tuple[torch.Tensor, torch.Tensor,
                                      torch.Tensor, torch.Tensor, float]:
    """Batch-level mixup: blend each sample with a random partner.

    lam ~ Beta(alpha, alpha), clipped to >= 0.5 so the first sample stays
    dominant (keeps the mixup "half-step" convention; avoids degenerate
    cases where targets invert).

    Teacher targets are convex-combined with the same lam — for global/
    spatial embeddings, L2 normalisation is applied downstream by the loss
    function, so we don't renormalise here.
    """
    if alpha <= 0:
        return mels, t_global, t_spatial, t_logits, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    lam = max(lam, 1.0 - lam)
    perm = torch.randperm(mels.size(0), device=mels.device)
    mels_m = lam * mels + (1.0 - lam) * mels[perm]
    tg_m = lam * t_global + (1.0 - lam) * t_global[perm]
    ts_m = lam * t_spatial + (1.0 - lam) * t_spatial[perm]
    tl_m = lam * t_logits + (1.0 - lam) * t_logits[perm]
    return mels_m, tg_m, ts_m, tl_m, lam


def build_noise_pool(segments_csv: str, audio_root_hint: str | None,
                     sr: int, segment_samples: int,
                     max_files: int = 100,
                     rng: np.random.Generator | None = None) -> NoisePool:
    """Build a NoisePool from unlabeled soundscape segments.

    Uses source_file paths from segments.csv; filters to unique files with
    source_type=='soundscape' and label_quality=='unlabeled'. Worker-level
    lazy loading means this constructor is cheap even for 100k+ candidates.
    """
    import pandas as pd
    df = pd.read_csv(segments_csv, low_memory=False,
                      usecols=["source_file", "source_type", "label_quality"])
    mask = ((df["source_type"].astype(str) == "soundscape")
            & (df["label_quality"].astype(str) == "unlabeled"))
    files = sorted(set(df.loc[mask, "source_file"].astype(str).tolist()))
    # Optionally filter to files that actually exist on disk
    existing = [f for f in files if Path(f).exists()]
    if not existing and audio_root_hint:
        # Some pipelines store relative paths; try prefixing the hint.
        base = Path(audio_root_hint)
        existing = [str(base / Path(f).name) for f in files
                    if (base / Path(f).name).exists()]
    return NoisePool(existing, sr=sr, segment_samples=segment_samples,
                     max_files=max_files, rng=rng)
