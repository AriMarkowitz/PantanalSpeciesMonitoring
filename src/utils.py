"""Shared utilities used by 2+ stage modules."""

import ast
import logging
import numpy as np
import soundfile as sf
from pathlib import Path


def load_audio_segment(path: str, start_sec: float, end_sec: float,
                       sr: int = 32000) -> np.ndarray:
    """Load a segment from an audio file without loading the full file.

    Returns mono float32 waveform resampled to `sr` if needed.
    """
    info = sf.info(str(path))
    file_sr = info.samplerate
    # Use native sr for frame offsets when reading, then resample
    start_frame = int(start_sec * file_sr)
    stop_frame = int(end_sec * file_sr)
    audio, _ = sf.read(path, start=start_frame, stop=stop_frame,
                       dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        import resampy
        audio = resampy.resample(audio, file_sr, sr)
    return audio


def get_audio_duration(path: str) -> float:
    """Get duration of an audio file in seconds (no full load)."""
    info = sf.info(str(path))
    return info.duration


def parse_secondary_labels(val: str) -> list[str]:
    """Parse secondary_labels column from train.csv.

    Handles formats: '[]', \"['sp1', 'sp2']\", \"['12345', '67890']\"
    """
    if not val or val == "[]":
        return []
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return []


def parse_soundscape_labels(label_str: str) -> list[str]:
    """Parse semicolon-separated labels from soundscape CSV."""
    if not label_str or label_str == "nan":
        return []
    return [s.strip() for s in str(label_str).split(";") if s.strip()]


def build_label_map(taxonomy_path: str) -> dict[str, int]:
    """Build {primary_label: index} mapping from taxonomy CSV."""
    import pandas as pd
    tax = pd.read_csv(taxonomy_path)
    labels = sorted(tax["primary_label"].astype(str).unique())
    return {label: i for i, label in enumerate(labels)}


def setup_logging(name: str, log_dir: str = None, level=logging.INFO) -> logging.Logger:
    """Configure logger with console + optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
