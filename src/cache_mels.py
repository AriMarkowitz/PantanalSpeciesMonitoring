"""Pre-cache mel spectrograms into the embeddings HDF5 file.

Computes mel spectrograms for all segments once and stores them as a new
dataset in embeddings.h5 (and optionally distill_embeddings.h5). This
eliminates the audio I/O bottleneck during student distillation training.

Usage:
    python src/cache_mels.py
    python src/cache_mels.py --distill   # also cache distill corpus
"""

import argparse
import sys
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from multiprocessing import Pool

from config import get_config
from utils import setup_logging, load_audio_segment

_IN_TTY = sys.stdout.isatty()


def _build_mel_basis(sr: int, mel_cfg: dict):
    """Build mel filterbank using librosa (one-time cost per process).

    Returns dict with numpy arrays for fast repeated use.
    """
    import librosa

    n_mels = mel_cfg.get("n_mels", 128)
    hop_length = int(sr * mel_cfg.get("hop_ms", 10) / 1000)
    win_length = int(sr * mel_cfg.get("win_ms", 25) / 1000)
    n_fft = max(512, win_length)
    fmin = mel_cfg.get("fmin", 60.0)
    fmax = mel_cfg.get("fmax", 16000.0)

    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                 fmin=fmin, fmax=fmax)  # (n_mels, n_fft//2+1)

    return {
        "mel_fb": mel_fb.astype(np.float32),
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
        "n_mels": n_mels,
    }

# Per-process cache for the mel basis
_mel_basis_cache = {}


def wav_to_mel_np(wav: np.ndarray, sr: int, mel_cfg: dict) -> np.ndarray:
    """Compute log-mel spectrogram using librosa.stft + cached mel filterbank.

    ~3ms per 5s segment at 32kHz. No torch/torchaudio dependency — runs
    identically on CPU inference (Kaggle) and GPU training nodes.

    Returns (n_mels, T) float32.
    """
    import librosa

    key = (sr, tuple(sorted(mel_cfg.items())))
    if key not in _mel_basis_cache:
        _mel_basis_cache[key] = _build_mel_basis(sr, mel_cfg)
    basis = _mel_basis_cache[key]

    target_len = int(sr * 5.0)
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    else:
        wav = wav[:target_len]

    # STFT → power spectrogram → mel → dB → [0,1]
    S = np.abs(librosa.stft(wav, n_fft=basis["n_fft"],
                            hop_length=basis["hop_length"],
                            win_length=basis["win_length"])) ** 2
    mel = basis["mel_fb"] @ S  # (n_mels, T_frames)

    # Log-scale with top_db clipping
    mel_db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    mel_db = np.maximum(mel_db, mel_db.max() - 80.0)

    # Normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype(np.float32)


def _process_segment(args):
    """Worker function for multiprocessing."""
    idx, source_file, start_sec, end_sec, sr, seg_dur, mel_cfg = args
    try:
        wav = load_audio_segment(source_file, start_sec, end_sec, sr=sr)
        mel = wav_to_mel_np(wav, sr, mel_cfg)
        return idx, mel, True
    except Exception:
        return idx, None, False


def cache_mels_for_h5(segments_csv: str, h5_path: str, cfg: dict, logger,
                      num_workers: int = 16):
    """Add a 'mel_spectrograms' dataset to an existing embeddings HDF5."""
    segments = pd.read_csv(segments_csv, low_memory=False)
    sr = cfg["data"]["sample_rate"]
    seg_dur = cfg["data"]["segment_duration"]
    mel_cfg = cfg.get("student_mel", {})

    h5 = h5py.File(h5_path, "a")  # append mode
    written = h5["written"][:]
    N = len(segments)

    if "mel_spectrograms" in h5:
        logger.info(f"mel_spectrograms already exists in {h5_path} — skipping")
        h5.close()
        return

    # Figure out mel shape from a test sample
    n_mels = mel_cfg.get("n_mels", 128)
    hop_length = int(sr * mel_cfg.get("hop_ms", 10) / 1000)
    target_len = int(sr * seg_dur)
    T = target_len // hop_length + 1
    logger.info(f"Mel shape per segment: ({n_mels}, {T}), dtype=float32")
    logger.info(f"Total storage: {N * n_mels * T * 4 / 1e9:.1f} GB")

    mel_ds = h5.create_dataset(
        "mel_spectrograms", shape=(N, n_mels, T), dtype="float32",
        chunks=(64, n_mels, T), compression="lzf",
    )

    # Build work items for valid segments only
    valid_indices = np.where(written)[0]
    logger.info(f"Computing mels for {len(valid_indices)}/{N} valid segments "
                f"with {num_workers} workers")

    work = []
    for idx in valid_indices:
        row = segments.iloc[idx]
        work.append((
            idx,
            str(row["source_file"]),
            float(row["start_sec"]),
            float(row["end_sec"]),
            sr, seg_dur, mel_cfg,
        ))

    # Process in chunks to limit memory and write progressively
    CHUNK = 5000
    n_done = 0
    n_failed = 0

    for chunk_start in range(0, len(work), CHUNK):
        chunk = work[chunk_start:chunk_start + CHUNK]

        with Pool(num_workers) as pool:
            results = pool.map(_process_segment, chunk)

        for idx, mel, ok in results:
            if ok and mel is not None:
                # Pad/trim to exact T
                if mel.shape[1] < T:
                    mel = np.pad(mel, ((0, 0), (0, T - mel.shape[1])))
                mel_ds[idx] = mel[:, :T]
            else:
                n_failed += 1

        n_done += len(chunk)
        logger.info(f"  {n_done}/{len(work)} segments processed "
                    f"({n_failed} failed)")
        for handler in logger.handlers:
            handler.flush()

    h5.flush()
    h5.close()
    logger.info(f"Cached {n_done - n_failed} mels to {h5_path}")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--distill", action="store_true",
                        help="Also cache distill corpus mels")
    parser.add_argument("--num-workers", type=int, default=16)
    args, _ = parser.parse_known_args()

    cfg = get_config()
    logger = setup_logging("cache_mels", cfg["outputs"]["logs_dir"])

    logger.info("Caching mel spectrograms into embeddings HDF5")

    # Primary corpus
    cache_mels_for_h5(
        cfg["outputs"]["segments_csv"],
        cfg["outputs"]["embeddings_h5"],
        cfg, logger, num_workers=args.num_workers,
    )

    # Distill corpus
    if args.distill:
        distill_h5 = cfg["outputs"].get("distill_embeddings_h5", "")
        distill_csv = cfg["outputs"].get("distill_segments_csv", "")
        if distill_h5 and Path(distill_h5).exists():
            cache_mels_for_h5(
                distill_csv, distill_h5, cfg, logger,
                num_workers=args.num_workers,
            )
        else:
            logger.warning("Distill HDF5 not found, skipping")

    logger.info("Done")


if __name__ == "__main__":
    main()
