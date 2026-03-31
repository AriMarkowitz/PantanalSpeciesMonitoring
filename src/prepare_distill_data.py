"""Stage 0b: Prepare distillation data — build distill_segments.csv.

Reads distill_manifest.csv (produced by scrape_distill_data.py), segments
each audio file into 5s chunks (same as Stage 0), deduplicates against the
existing segments.csv (by sha1 in the manifest and by audio file basename),
and labels any segment whose species matches the project taxonomy.

Outputs:
    outputs/distill_segments.csv  — same 11-column schema as segments.csv

Usage:
    python src/prepare_distill_data.py
    python src/prepare_distill_data.py --set stage0.skip_silence_filter=true
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import get_config
from utils import get_audio_duration, setup_logging
from prepare_data import segment_file, check_energy


# ── Label matching ──────────────────────────────────────────────────────────

def build_sci_name_to_label(taxonomy_path: str) -> dict[str, str]:
    """Return {lower(scientific_name): primary_label} from taxonomy CSV."""
    tax = pd.read_csv(taxonomy_path)
    return {
        row["scientific_name"].strip().lower(): str(row["primary_label"])
        for _, row in tax.iterrows()
    }


# ── Deduplication ────────────────────────────────────────────────────────────

def load_existing_dedup(segments_csv: str, distill_manifest: str) -> tuple[set, set]:
    """Return (known_sha1s, known_basenames) from existing segments + manifest.

    segments.csv has no sha1 column, so we dedup on audio file basename.
    distill_manifest sha1 field deduplicates within the distill set itself.
    """
    known_basenames: set[str] = set()
    known_sha1s: set[str] = set()

    if os.path.exists(segments_csv):
        seg = pd.read_csv(segments_csv, low_memory=False)
        for p in seg["source_file"].dropna().unique():
            known_basenames.add(Path(p).name)

    # Also include basenames from any prior distill_segments.csv (resumability)
    distill_seg_csv = str(segments_csv).replace("segments.csv", "distill_segments.csv")
    if os.path.exists(distill_seg_csv):
        try:
            dseg = pd.read_csv(distill_seg_csv, low_memory=False)
            for p in dseg["source_file"].dropna().unique():
                known_basenames.add(Path(p).name)
        except pd.errors.EmptyDataError:
            pass  # empty file from a prior failed run — ignore

    return known_sha1s, known_basenames


# ── Main processing ──────────────────────────────────────────────────────────

def process_distill_manifest(
    manifest_path: str,
    audio_base_dir: str,
    sci_to_label: dict[str, str],
    existing_sha1s: set[str],
    existing_basenames: set[str],
    seg_dur: float,
    overlap: float,
    sr: int,
    skip_silence_filter: bool,
    energy_threshold: float,
    min_active_fraction: float,
    logger,
) -> pd.DataFrame:
    """Segment distill audio files and return a segments DataFrame."""
    manifest = pd.read_csv(manifest_path, low_memory=False)
    logger.info(f"Loaded distill manifest: {len(manifest)} files")

    # Track dedup counters for logging
    skipped_sha1 = 0
    skipped_basename = 0
    skipped_missing = 0
    skipped_silence = 0

    rows = []
    seen_sha1s: set[str] = set(existing_sha1s)
    seen_basenames: set[str] = set(existing_basenames)

    for _, mrow in tqdm(manifest.iterrows(), total=len(manifest),
                        desc="prepare_distill"):
        # Resolve absolute audio path
        rel_path = str(mrow.get("relative_path", ""))
        audio_path = os.path.join(audio_base_dir, rel_path)

        if not os.path.exists(audio_path):
            skipped_missing += 1
            continue

        basename = Path(audio_path).name

        # Dedup: sha1 (exact duplicate audio content)
        sha1 = str(mrow.get("sha1", "")) if pd.notna(mrow.get("sha1")) else ""
        if sha1 and sha1 in seen_sha1s:
            skipped_sha1 += 1
            continue
        # Dedup: basename (file already in train_audio or a previous distill run)
        if basename in seen_basenames:
            skipped_basename += 1
            continue

        seen_sha1s.add(sha1)
        seen_basenames.add(basename)

        # Determine label
        sci_name = str(mrow.get("species_name", "")).strip().lower()
        primary_label = sci_to_label.get(sci_name, "")
        is_labeled = bool(primary_label)
        label_quality = "strong_primary" if is_labeled else "unlabeled"

        # Get duration and segment
        try:
            duration = get_audio_duration(audio_path)
        except Exception as e:
            logger.warning(f"Could not read duration for {audio_path}: {e}")
            skipped_missing += 1
            continue

        source_type = str(mrow.get("source", "distill")).lower()
        segments = segment_file(audio_path, duration, seg_dur, overlap)

        # event_start_sec from manifest: timestamp of the first vocalization.
        # Passed through to distill_segments.csv so the embedder can center
        # its 9s context window on the vocalization rather than the segment midpoint.
        raw_event = mrow.get("event_start_sec")
        try:
            event_start_sec = float(raw_event) if raw_event != "" and pd.notna(raw_event) else None
        except (TypeError, ValueError):
            event_start_sec = None

        for start, end in segments:
            # Silence filter (optional)
            if not skip_silence_filter:
                try:
                    if not check_energy(audio_path, start, end, sr,
                                        energy_threshold, min_active_fraction):
                        skipped_silence += 1
                        continue
                except Exception:
                    pass  # If we can't check energy, include the segment

            seg_id = f"distill/{basename}_{start:.1f}_{end:.1f}"
            rows.append({
                "segment_id": seg_id,
                "source_file": audio_path,
                "start_sec": start,
                "end_sec": end,
                "source_type": source_type,
                "primary_label": primary_label,
                "secondary_labels": "",
                "is_labeled": is_labeled,
                "label_quality": label_quality,
                "event_start_sec": event_start_sec if event_start_sec is not None else "",
            })

    logger.info(
        f"Dedup: skipped {skipped_sha1} sha1 dups, "
        f"{skipped_basename} basename dups, "
        f"{skipped_missing} missing files"
    )
    if not skip_silence_filter:
        logger.info(f"Silence filter removed {skipped_silence} segments")

    return pd.DataFrame(rows)


# ── Entry point ──────────────────────────────────────────────────────────────

def main(cfg: dict):
    logger = setup_logging("prepare_distill_data", cfg["outputs"]["logs_dir"])
    logger.info("Stage 0b: Distill data preparation")

    manifest_path = cfg["data"].get(
        "distill_manifest_csv", "data/distill_manifest.csv"
    )
    audio_base_dir = cfg["data"].get(
        "distill_audio_dir", "data/distill_audio"
    )
    taxonomy_path = cfg["data"]["taxonomy_csv"]
    segments_csv = cfg["outputs"]["segments_csv"]
    out_path = cfg["outputs"].get(
        "distill_segments_csv", "outputs/distill_segments.csv"
    )

    seg_dur = cfg["data"]["segment_duration"]
    overlap = cfg["data"]["segment_overlap"]
    sr = cfg["data"]["sample_rate"]
    skip_silence = cfg["stage0"].get("skip_silence_filter", False)
    energy_threshold = cfg["stage0"]["energy_threshold"]
    min_active = cfg["stage0"]["min_active_fraction"]

    # Build label lookup
    sci_to_label = build_sci_name_to_label(taxonomy_path)
    logger.info(f"Taxonomy label map: {len(sci_to_label)} species")

    # Load dedup sets
    existing_sha1s, existing_basenames = load_existing_dedup(
        segments_csv, manifest_path
    )
    logger.info(
        f"Dedup index: {len(existing_basenames)} known basenames, "
        f"{len(existing_sha1s)} known sha1s"
    )

    # Process manifest
    distill_df = process_distill_manifest(
        manifest_path=manifest_path,
        audio_base_dir=audio_base_dir,
        sci_to_label=sci_to_label,
        existing_sha1s=existing_sha1s,
        existing_basenames=existing_basenames,
        seg_dur=seg_dur,
        overlap=overlap,
        sr=sr,
        skip_silence_filter=skip_silence,
        energy_threshold=energy_threshold,
        min_active_fraction=min_active,
        logger=logger,
    )

    # Summary
    n_labeled = int(distill_df["is_labeled"].sum()) if len(distill_df) else 0
    n_total = len(distill_df)
    logger.info(f"Total distill segments: {n_total} ({n_labeled} labeled)")
    if n_total:
        for st in distill_df["source_type"].unique():
            sub = distill_df[distill_df["source_type"] == st]
            lab = sub["is_labeled"].sum()
            logger.info(f"  {st}: {len(sub)} segments ({lab} labeled)")

    # Write output
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    distill_df.to_csv(out_path, index=False)
    logger.info(f"Saved {n_total} distill segments to {out_path}")


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
