"""Stage 0: Data preparation — build segments.csv from raw data.

Reads train.csv (XC/iNat recordings) and train_soundscapes_labels.csv,
generates 5s segment metadata with label provenance tracking.

No audio files are created — segments are defined by (source_file, start_sec, end_sec)
and loaded on-the-fly by downstream stages.

Usage:
    python src/prepare_data.py
    python src/prepare_data.py --set stage0.skip_silence_filter=true
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import get_config
from utils import (
    get_audio_duration,
    parse_secondary_labels,
    parse_soundscape_labels,
    load_audio_segment,
    setup_logging,
)


def segment_file(path: str, duration: float, seg_dur: float,
                 overlap: float) -> list[tuple[float, float]]:
    """Generate (start_sec, end_sec) tuples for a file."""
    step = seg_dur - overlap
    segments = []
    start = 0.0
    while start + seg_dur <= duration + 0.01:  # tolerance for float rounding
        end = min(start + seg_dur, duration)
        segments.append((round(start, 3), round(end, 3)))
        start += step
    return segments


def check_energy(path: str, start_sec: float, end_sec: float,
                 sr: int, threshold: float, min_active: float) -> bool:
    """Return True if the segment has sufficient acoustic energy."""
    audio = load_audio_segment(path, start_sec, end_sec, sr)
    # Frame-level energy (10ms frames)
    frame_len = sr // 100
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return False
    frames = audio[:n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    active_fraction = np.mean(rms > threshold)
    return active_fraction >= min_active


def process_train_audio(cfg: dict, label_map: dict,
                        logger) -> pd.DataFrame:
    """Generate segments for XC/iNat focal recordings."""
    train_csv = cfg["data"]["train_csv"]
    audio_dir = cfg["data"]["train_audio_dir"]
    seg_dur = cfg["data"]["segment_duration"]
    overlap = cfg["data"]["segment_overlap"]
    sr = cfg["data"]["sample_rate"]

    df = pd.read_csv(train_csv)
    logger.info(f"Processing {len(df)} train_audio files")

    rows = []
    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="train_audio"):
        path = os.path.join(audio_dir, row["filename"])
        if not os.path.exists(path):
            skipped += 1
            continue

        duration = get_audio_duration(path)
        primary = str(row["primary_label"])
        secondaries = parse_secondary_labels(str(row["secondary_labels"]))
        source = row.get("collection", "unknown")

        segments = segment_file(path, duration, seg_dur, overlap)
        for start, end in segments:
            seg_id = f"{row['filename']}_{start:.1f}_{end:.1f}"
            rows.append({
                "segment_id": seg_id,
                "source_file": path,
                "start_sec": start,
                "end_sec": end,
                "source_type": source.lower(),
                "primary_label": primary,
                "secondary_labels": ";".join(secondaries),
                "is_labeled": True,
                "label_quality": "strong_primary",
            })

    if skipped:
        logger.warning(f"Skipped {skipped} missing audio files")
    logger.info(f"Generated {len(rows)} segments from train_audio")
    return pd.DataFrame(rows)


def process_soundscapes(cfg: dict, label_map: dict,
                        logger) -> pd.DataFrame:
    """Generate segments for soundscape files (labeled + unlabeled)."""
    sc_dir = cfg["data"]["soundscape_dir"]
    sc_labels_csv = cfg["data"]["soundscape_labels_csv"]
    seg_dur = cfg["data"]["segment_duration"]
    sr = cfg["data"]["sample_rate"]

    # Load soundscape labels
    sc_labels = pd.read_csv(sc_labels_csv)
    # Parse time columns to seconds
    def time_to_sec(t: str) -> float:
        parts = str(t).split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(t)

    sc_labels["start_sec"] = sc_labels["start"].apply(time_to_sec)
    sc_labels["end_sec"] = sc_labels["end"].apply(time_to_sec)

    # Build lookup: (filename, start_sec) -> labels
    label_lookup = {}
    for _, row in sc_labels.iterrows():
        key = (row["filename"], row["start_sec"])
        labels = parse_soundscape_labels(str(row["primary_label"]))
        label_lookup[key] = labels

    labeled_files = set(sc_labels["filename"].unique())

    # Process all soundscape files
    all_files = sorted(os.listdir(sc_dir))
    logger.info(f"Processing {len(all_files)} soundscape files "
                f"({len(labeled_files)} labeled)")

    rows = []
    for fname in tqdm(all_files, desc="soundscapes"):
        path = os.path.join(sc_dir, fname)
        if not os.path.isfile(path):
            continue

        duration = get_audio_duration(path)
        is_labeled_file = fname in labeled_files

        segments = segment_file(path, duration, seg_dur, overlap=0.0)
        for start, end in segments:
            seg_id = f"{fname}_{start:.1f}_{end:.1f}"
            key = (fname, start)

            if is_labeled_file and key in label_lookup:
                labels = label_lookup[key]
                rows.append({
                    "segment_id": seg_id,
                    "source_file": path,
                    "start_sec": start,
                    "end_sec": end,
                    "source_type": "soundscape",
                    "primary_label": ";".join(labels),
                    "secondary_labels": "",
                    "is_labeled": True,
                    "label_quality": "strong_multilabel",
                })
            else:
                rows.append({
                    "segment_id": seg_id,
                    "source_file": path,
                    "start_sec": start,
                    "end_sec": end,
                    "source_type": "soundscape",
                    "primary_label": "",
                    "secondary_labels": "",
                    "is_labeled": False,
                    "label_quality": "unlabeled",
                })

    logger.info(f"Generated {len(rows)} segments from soundscapes "
                f"({sum(1 for r in rows if r['is_labeled'])} labeled)")
    return pd.DataFrame(rows)


def filter_silence(df: pd.DataFrame, cfg: dict, logger) -> pd.DataFrame:
    """Optionally filter segments with insufficient acoustic energy."""
    if cfg["stage0"].get("skip_silence_filter", False):
        logger.info("Silence filtering skipped (skip_silence_filter=true)")
        return df

    sr = cfg["data"]["sample_rate"]
    threshold = cfg["stage0"]["energy_threshold"]
    min_active = cfg["stage0"]["min_active_fraction"]

    logger.info(f"Filtering silence (threshold={threshold}, "
                f"min_active={min_active})")

    keep = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="silence filter"):
        if check_energy(row["source_file"], row["start_sec"], row["end_sec"],
                        sr, threshold, min_active):
            keep.append(True)
        else:
            keep.append(False)

    before = len(df)
    df = df[keep].reset_index(drop=True)
    logger.info(f"Silence filter: {before} → {len(df)} segments "
                f"({before - len(df)} removed)")
    return df


def main(cfg: dict):
    logger = setup_logging("prepare_data", cfg["outputs"]["logs_dir"])
    logger.info("Stage 0: Data preparation")

    label_map = {}  # built later from taxonomy, not needed for segmentation

    # Process both data sources
    train_df = process_train_audio(cfg, label_map, logger)
    sc_df = process_soundscapes(cfg, label_map, logger)

    # Combine
    segments = pd.concat([train_df, sc_df], ignore_index=True)
    logger.info(f"Total segments before filtering: {len(segments)}")

    # Silence filter (optional, slow on full dataset)
    segments = filter_silence(segments, cfg, logger)

    # Summary stats
    logger.info("Segment summary:")
    logger.info(f"  Total: {len(segments)}")
    for st in segments["source_type"].unique():
        sub = segments[segments["source_type"] == st]
        labeled = sub["is_labeled"].sum()
        logger.info(f"  {st}: {len(sub)} ({labeled} labeled)")

    # Write output
    out_path = cfg["outputs"]["segments_csv"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    segments.to_csv(out_path, index=False)
    logger.info(f"Saved {len(segments)} segments to {out_path}")


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
