"""Collect filtered distillation audio from iNat Sounds and BirdSet.

This script is designed for storage-constrained workflows:
- builds/updates a manifest CSV for reproducibility
- filters aggressively (taxa, geography, duration, max samples)
- downloads only selected clips
- deduplicates against existing manifests and local files

Primary source support:
  1) iNat Sounds 2024 (streaming extract from tar archives)
  2) BirdSet (optional; via Hugging Face datasets package)

Examples:
  # iNat only, Pantanal bbox
    python src/scrape_distill_data.py \
    --output-dir data/distill_audio \
    --manifest-path data/distill_manifest.csv \
    --inat --inat-max-files 30000 \
    --inat-supercategories aves,amphibia,insecta,mammalia,reptilia \
    --bbox=-22,-62,-10,-44 --skip-existing

  # BirdSet geographic streaming — all 10 configs, Pantanal bbox, quality filter
  python src/scrape_distill_data.py \
    --output-dir data/distill_audio \
    --manifest-path data/distill_manifest.csv \
    --birdset-geo \
    --birdset-configs PER,NES,UHH,HSN,NBP,POW,SSW,SNE,XCM,XCL \
    --birdset-geo-max-per-config 5000 \
    --train-csv data/train.csv \
    --skip-existing

  # Add BirdSet PER subset (non-streaming, legacy)
  python src/scrape_distill_data.py \
    --output-dir data/distill_audio \
    --manifest-path data/distill_manifest.csv \
    --birdset --birdset-config PER --birdset-split train --birdset-max-files 3000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
from tqdm import tqdm


EBIRD_TAXONOMY_URL = (
    "https://www.birds.cornell.edu/clementschecklist/wp-content/uploads/2024/10/"
    "eBird_taxonomy_v2024.csv"
)

INAT_BASE = "https://ml-inat-competition-datasets.s3.amazonaws.com/sounds/2024"
INAT_AUDIO_URLS = {
    "train": f"{INAT_BASE}/train.tar.gz",
    "val": f"{INAT_BASE}/val.tar.gz",
    "test": f"{INAT_BASE}/test.tar.gz",
}
INAT_META_URLS = {
    "train": f"{INAT_BASE}/train.json.tar.gz",
    "val": f"{INAT_BASE}/val.json.tar.gz",
    "test": f"{INAT_BASE}/test.json.tar.gz",
}


MANIFEST_COLUMNS = [
    "source",
    "source_split",
    "source_id",
    "species_name",
    "common_name",
    "supercategory",
    "latitude",
    "longitude",
    "duration_sec",
    "sample_rate",
    "relative_path",
    "sha1",
]


@dataclass
class DedupIndex:
    source_ids: set[str]
    hashes: set[str]
    basenames: set[str]


def setup_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("scrape_distill_data")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    return logger


def ensure_manifest(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()


def load_dedup_index(manifest_path: Path, output_dir: Path) -> DedupIndex:
    source_ids: set[str] = set()
    hashes: set[str] = set()
    basenames: set[str] = set()

    if manifest_path.exists():
        with manifest_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("source_id", "")
                sha = row.get("sha1", "")
                rel = row.get("relative_path", "")
                if sid:
                    source_ids.add(sid)
                if sha:
                    hashes.add(sha)
                if rel:
                    basenames.add(Path(rel).name)

    if output_dir.exists():
        for p in output_dir.rglob("*.wav"):
            basenames.add(p.name)

    return DedupIndex(source_ids=source_ids, hashes=hashes, basenames=basenames)


def append_manifest_rows(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        for row in rows:
            writer.writerow(row)


def _download_small_file(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def _load_inat_json(split: str, logger: logging.Logger) -> dict[str, Any]:
    meta_url = INAT_META_URLS[split]
    logger.info(f"Downloading iNat metadata for split={split}: {meta_url}")
    raw = _download_small_file(meta_url)
    fileobj = io.BytesIO(raw)
    with tarfile.open(fileobj=fileobj, mode="r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".json")]
        if not members:
            raise RuntimeError(f"No JSON found in {meta_url}")
        extracted = tf.extractfile(members[0])
        if extracted is None:
            raise RuntimeError(f"Failed to extract JSON from {meta_url}")
        return json.load(extracted)


def _parse_bbox(bbox: str | None):
    if not bbox:
        return None
    parts = [float(x.strip()) for x in bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("--bbox must be min_lat,min_lon,max_lat,max_lon")
    min_lat, min_lon, max_lat, max_lon = parts
    return min_lat, min_lon, max_lat, max_lon


def _in_bbox(lat: Any, lon: Any, bbox: tuple[float, float, float, float] | None) -> bool:
    if bbox is None:
        return True
    if lat is None or lon is None:
        return False
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return False
    min_lat, min_lon, max_lat, max_lon = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def _sha1_path(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_float(x: Any):
    if x is None:
        return ""
    try:
        return float(x)
    except (TypeError, ValueError):
        return ""


def load_ebird_taxonomy(cache_dir: Path, logger: logging.Logger) -> dict[str, dict[str, str]]:
    """Load eBird taxonomy CSV.

    Search order:
      1. Any ebird_taxonomy*.csv / eBird_taxonomy*.csv in cache_dir
      2. data/ directory relative to cwd (where user may have placed it manually)
      3. Download from Cornell (may be blocked on HPC — use local copy instead)

    Returns a dict: ebird_code (lowercase) -> {"scientific_name": ..., "common_name": ...}
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect candidate paths: cache_dir first, then data/
    search_dirs = [cache_dir, Path("data")]
    cache_path = None
    for d in search_dirs:
        for p in sorted(d.glob("*[eE][bB]ird_taxonomy*.csv"), reverse=True):
            cache_path = p
            break
        if cache_path:
            break

    if cache_path and cache_path.exists():
        logger.info(f"Using local eBird taxonomy: {cache_path}")
    else:
        dest = cache_dir / "ebird_taxonomy_v2024.csv"
        logger.info(f"Downloading eBird taxonomy → {dest}")
        data = _download_small_file(EBIRD_TAXONOMY_URL)
        dest.write_bytes(data)
        cache_path = dest

    mapping: dict[str, dict[str, str]] = {}
    with cache_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Column names vary slightly between versions; try both
            code = (row.get("species_code") or row.get("SPECIES_CODE") or "").strip().lower()
            sci = (row.get("SCI_NAME") or row.get("sci_name") or row.get("SCIENTIFIC NAME") or "").strip()
            common = (row.get("PRIMARY_COM_NAME") or row.get("primary_com_name") or row.get("COMMON NAME") or "").strip()
            if code:
                mapping[code] = {"scientific_name": sci, "common_name": common}

    logger.info(f"Loaded eBird taxonomy: {len(mapping)} species")
    return mapping


def build_birdset_label_map(config_name: str, split: str, logger: logging.Logger) -> dict[int, str]:
    """Return {int_index: ebird_code_string} for a BirdSet config using its ClassLabel feature."""
    try:
        from datasets import load_dataset_builder
    except ImportError:
        return {}

    try:
        builder = load_dataset_builder("DBD-research-group/BirdSet", config_name, trust_remote_code=True)
        builder.download_and_prepare()
        features = builder.info.features
        ebird_feature = features.get("ebird_code")
        if ebird_feature is not None and hasattr(ebird_feature, "names"):
            result = {i: name for i, name in enumerate(ebird_feature.names)}
            logger.info(f"BirdSet {config_name} label map: {len(result)} classes")
            return result
    except Exception as e:
        logger.warning(f"Could not load BirdSet label map for {config_name}: {e}")
    return {}


def collect_inat(
    output_dir: Path,
    manifest_path: Path,
    dedup: DedupIndex,
    supercategories: set[str],
    max_files: int,
    bbox: tuple[float, float, float, float] | None,
    min_duration: float,
    include_splits: list[str],
    skip_existing: bool,
    logger: logging.Logger,
):
    output_base = output_dir / "inat_sounds_2024"
    output_base.mkdir(parents=True, exist_ok=True)

    target_by_split: dict[str, dict[str, dict[str, Any]]] = {s: {} for s in include_splits}

    logger.info("Building filtered iNat candidate list from metadata")
    selected_count = 0
    for split in include_splits:
        meta = _load_inat_json(split, logger)
        audio_by_id = {a["id"]: a for a in meta.get("audio", [])}
        cat_by_id = {c["id"]: c for c in meta.get("categories", [])}

        for ann in meta.get("annotations", []):
            if selected_count >= max_files:
                break
            audio = audio_by_id.get(ann.get("audio_id"))
            cat = cat_by_id.get(ann.get("category_id"))
            if audio is None or cat is None:
                continue

            sup = str(cat.get("supercategory", "")).lower()
            if supercategories and sup not in supercategories:
                continue

            duration = float(audio.get("duration") or 0.0)
            if duration < min_duration:
                continue

            lat = audio.get("latitude")
            lon = audio.get("longitude")
            if not _in_bbox(lat, lon, bbox):
                continue

            file_name = str(audio.get("file_name", ""))
            if not file_name:
                continue

            source_id = f"inat:{split}:{audio.get('id')}"
            basename = Path(file_name).name

            if skip_existing:
                if source_id in dedup.source_ids or basename in dedup.basenames:
                    continue

            target_by_split[split][file_name] = {
                "source": "inat",
                "source_split": split,
                "source_id": source_id,
                "species_name": str(cat.get("name", "")),
                "common_name": str(cat.get("common_name", "")),
                "supercategory": sup,
                "latitude": _safe_float(lat),
                "longitude": _safe_float(lon),
                "duration_sec": duration,
            }
            selected_count += 1

        logger.info(f"iNat candidates after split={split}: {selected_count}")
        if selected_count >= max_files:
            break

    if selected_count == 0:
        logger.warning("No iNat candidates matched filters")
        return

    logger.info(f"Extracting up to {selected_count} iNat files from streaming tar archives")
    new_rows: list[dict[str, Any]] = []
    extracted_total = 0

    for split in include_splits:
        wanted = target_by_split.get(split, {})
        if not wanted:
            continue

        split_out = output_base / split
        split_out.mkdir(parents=True, exist_ok=True)

        url = INAT_AUDIO_URLS[split]
        logger.info(f"Streaming split={split}: {url}")
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            with tarfile.open(fileobj=resp, mode="r|gz") as tf:
                pbar = tqdm(total=len(wanted), desc=f"inat-{split}")
                for member in tf:
                    if not member.isfile():
                        continue
                    rel_name = member.name
                    if rel_name not in wanted:
                        continue

                    meta = wanted[rel_name]
                    source_id = meta["source_id"]
                    source_basename = Path(rel_name).name
                    out_name = f"{source_id.replace(':', '_')}_{source_basename}"
                    out_path = split_out / out_name

                    extracted = tf.extractfile(member)
                    if extracted is None:
                        continue

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(extracted.read())
                        tmp_path = Path(tmp.name)

                    try:
                        if out_path.exists() and skip_existing:
                            tmp_path.unlink(missing_ok=True)
                            pbar.update(1)
                            continue

                        sha1 = _sha1_path(tmp_path)
                        if skip_existing and sha1 in dedup.hashes:
                            tmp_path.unlink(missing_ok=True)
                            pbar.update(1)
                            continue

                        shutil.move(str(tmp_path), str(out_path))
                        info = sf.info(str(out_path))

                        rel_out = out_path.relative_to(output_dir)
                        row = {
                            "source": meta["source"],
                            "source_split": meta["source_split"],
                            "source_id": source_id,
                            "species_name": meta["species_name"],
                            "common_name": meta["common_name"],
                            "supercategory": meta["supercategory"],
                            "latitude": meta["latitude"],
                            "longitude": meta["longitude"],
                            "duration_sec": float(info.duration),
                            "sample_rate": int(info.samplerate),
                            "relative_path": str(rel_out),
                            "sha1": sha1,
                        }
                        new_rows.append(row)
                        dedup.source_ids.add(source_id)
                        dedup.hashes.add(sha1)
                        dedup.basenames.add(out_path.name)
                        extracted_total += 1
                    finally:
                        tmp_path.unlink(missing_ok=True)
                    pbar.update(1)
                    if extracted_total >= max_files:
                        break
                pbar.close()

        if extracted_total >= max_files:
            break

    append_manifest_rows(manifest_path, new_rows)
    logger.info(f"iNat complete: extracted={extracted_total}, manifest_rows_added={len(new_rows)}")


def collect_birdset(
    output_dir: Path,
    manifest_path: Path,
    dedup: DedupIndex,
    config_name: str,
    split: str,
    max_files: int,
    species_allowlist_path: Path | None,
    ebird_taxonomy: dict[str, dict[str, str]],
    skip_existing: bool,
    logger: logging.Logger,
):
    try:
        from datasets import load_dataset
    except Exception as e:
        logger.error(
            "BirdSet requested but Hugging Face datasets package is unavailable. "
            "Install with: pip install datasets<=3.6.0"
        )
        raise RuntimeError("Missing optional dependency: datasets") from e

    output_base = output_dir / "birdset"
    output_base.mkdir(parents=True, exist_ok=True)

    allowlist: set[str] | None = None
    if species_allowlist_path and species_allowlist_path.exists():
        allowlist = {
            line.strip()
            for line in species_allowlist_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        logger.info(f"Loaded BirdSet species allowlist: {len(allowlist)} entries")

    # Build int→ebird_code map from the dataset's ClassLabel feature
    int2ebird = build_birdset_label_map(config_name, split, logger)

    logger.info(f"Loading BirdSet dataset stream: config={config_name}, split={split}")
    # Load raw — do NOT use cast_column/Audio decode; it silently returns array=None
    # when librosa resampling fails in the isolated venv. We decode bytes ourselves.
    ds = load_dataset("DBD-research-group/BirdSet", config_name, split=split, streaming=True, trust_remote_code=True)

    new_rows: list[dict[str, Any]] = []
    count = 0
    target_sr = 32000

    for ex in tqdm(ds, total=max_files, desc=f"birdset-{config_name}-{split}"):
        if count >= max_files:
            break

        audio_field = ex.get("audio")
        if not isinstance(audio_field, dict):
            continue
        raw_bytes = audio_field.get("bytes")
        if not raw_bytes:
            continue

        # Resolve ebird_code: may be int index or string depending on HF version
        raw_ebird = ex.get("ebird_code")
        if isinstance(raw_ebird, int) and int2ebird:
            ebird_code = int2ebird.get(raw_ebird, "")
        else:
            ebird_code = str(raw_ebird or "").lower()

        if allowlist is not None:
            if not ebird_code or ebird_code not in allowlist:
                continue

        # Look up scientific name and common name from eBird taxonomy
        tax = ebird_taxonomy.get(ebird_code, {})
        scientific_name = tax.get("scientific_name", "")
        common_name = tax.get("common_name", "")

        # Decode bytes → numpy array via soundfile
        try:
            arr, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
        except Exception as e:
            logger.warning(f"soundfile decode failed for example {count}: {e}")
            continue
        if arr.ndim > 1:
            arr = arr.mean(axis=1)

        # Resample to 32kHz if needed
        if sr != target_sr:
            try:
                import resampy
                arr = resampy.resample(arr, sr, target_sr)
            except ImportError:
                from scipy.signal import resample_poly
                import math
                g = math.gcd(target_sr, sr)
                arr = resample_poly(arr, target_sr // g, sr // g).astype("float32")
            sr = target_sr

        # Read lat/lon — BirdSet exposes these in streaming mode
        lat = _safe_float(ex.get("lat") or ex.get("latitude"))
        lon = _safe_float(ex.get("long") or ex.get("longitude"))

        filepath = str(ex.get("filepath", ""))
        source_id = f"birdset:{config_name}:{split}:{filepath or count}"
        basename = Path(filepath).stem if filepath else f"sample_{count}"
        out_name = f"birdset_{config_name}_{split}_{basename}.wav"

        out_path = output_base / config_name / split / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if skip_existing and (source_id in dedup.source_ids or out_path.name in dedup.basenames):
            continue
        sf.write(str(out_path), arr, sr)
        sha1 = _sha1_path(out_path)

        if skip_existing and sha1 in dedup.hashes:
            out_path.unlink(missing_ok=True)
            continue

        rel_out = out_path.relative_to(output_dir)
        row = {
            "source": "birdset",
            "source_split": split,
            "source_id": source_id,
            "species_name": scientific_name or ebird_code,
            "common_name": common_name,
            "supercategory": "aves",
            "latitude": lat,
            "longitude": lon,
            "duration_sec": float(len(arr) / sr) if sr > 0 else "",
            "sample_rate": sr,
            "relative_path": str(rel_out),
            "sha1": sha1,
        }
        new_rows.append(row)
        dedup.source_ids.add(source_id)
        dedup.hashes.add(sha1)
        dedup.basenames.add(out_path.name)
        count += 1

    append_manifest_rows(manifest_path, new_rows)
    logger.info(f"BirdSet complete: extracted={count}, manifest_rows_added={len(new_rows)}")


def load_train_xc_ids(train_csv_path: Path | None, logger: logging.Logger) -> set[str]:
    """Parse XC IDs from the competition train.csv filename column.

    BirdSet source_ids look like 'XC123456'. We strip the 'XC' prefix and
    store bare integer strings for fast lookup.

    Returns a set of XC id strings (bare integers, e.g. {"123456", "78901"}).
    """
    if train_csv_path is None or not train_csv_path.exists():
        logger.info("No train.csv provided or not found — skipping XC ID dedup")
        return set()

    xc_ids: set[str] = set()
    with train_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fname_col = next(
            (c for c in (reader.fieldnames or []) if c.lower() in ("filename", "file_name")),
            None,
        )
        if fname_col is None:
            logger.warning("train.csv has no 'filename' column — skipping XC ID dedup")
            return set()
        for row in reader:
            fname = Path(row[fname_col]).stem  # e.g. "XC123456" or "XC123456.ogg"
            bare = fname.upper().lstrip("XC").lstrip("0") or "0"
            xc_ids.add(bare)
            # also store with leading zeros stripped and the raw form
            xc_ids.add(fname.upper().replace("XC", "").replace("xc", ""))

    logger.info(f"Loaded {len(xc_ids)} XC IDs from train.csv for dedup")
    return xc_ids


def _extract_xc_id(filepath: str) -> str | None:
    """Extract the bare XC integer string from a BirdSet filepath field.

    BirdSet filepaths look like: 'XC123456.ogg', 'xc123456', or just '123456'.
    Returns the bare integer string (stripped of 'XC' prefix and leading zeros).
    """
    if not filepath:
        return None
    stem = Path(filepath).stem.upper()
    bare = stem.lstrip("XC").lstrip("0") or "0"
    return bare


def collect_birdset_streaming_geo(
    output_dir: Path,
    manifest_path: Path,
    dedup: DedupIndex,
    configs: list[str],
    split: str,
    bbox: tuple[float, float, float, float],
    max_per_config: int,
    require_detected_events: bool,
    ebird_taxonomy: dict[str, dict[str, str]],
    train_xc_ids: set[str],
    skip_existing: bool,
    logger: logging.Logger,
    no_bbox_configs: set[str] | None = None,
):
    """Stream all BirdSet configs with geographic bbox filter + quality filter.

    Key differences from collect_birdset():
    - Streams all configs; no full dataset download
    - Geographic filter applied per-example before any decode
    - Pre-embed XC ID dedup: skip examples whose XC ID is in train.csv
    - Quality filter: prefer clips with detected_events (vocalization timestamps)
      when require_detected_events=True; fall back to all clips otherwise
    - No random per-species cap — keeps all geographically-relevant clips

    Args:
        bbox: (min_lat, min_lon, max_lat, max_lon)
        require_detected_events: if True, skip clips with empty detected_events
        train_xc_ids: set of bare XC integer strings from train.csv
        no_bbox_configs: config names that skip the bbox filter entirely
            (e.g. {"PER", "NES"} — region-specific datasets where every clip
            is relevant regardless of coordinates)
    """
    if no_bbox_configs is None:
        no_bbox_configs = set()
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("Missing optional dependency: datasets. pip install datasets") from e

    # Route HF cache to /tmp to avoid filling home quota with shard downloads
    import os as _os
    hf_cache = _os.environ.get("HF_DATASETS_CACHE") or _os.environ.get("HF_HOME")
    if not hf_cache:
        _os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_datasets_cache"
        logger.info("Routing HF datasets cache → /tmp/hf_datasets_cache")

    output_base = output_dir / "birdset"
    output_base.mkdir(parents=True, exist_ok=True)
    target_sr = 32000
    seg_dur = 5.0  # seconds — extract one 5s window per clip

    total_written = 0

    for config_name in configs:
        logger.info(f"Streaming BirdSet {config_name}/{split} with geo filter bbox={bbox}")
        try:
            ds = load_dataset(
                "DBD-research-group/BirdSet", config_name,
                split=split, streaming=True, trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(f"Could not load BirdSet {config_name}: {e} — skipping")
            continue

        # Build int→ebird_code map for this config
        int2ebird = build_birdset_label_map(config_name, split, logger)

        new_rows: list[dict] = []
        count = 0
        skip_geo = config_name in no_bbox_configs
        if skip_geo:
            logger.info(f"  {config_name}: bbox filter disabled (region-specific config)")

        for ex in tqdm(ds, desc=f"birdset-{config_name}", disable=not _os.isatty(1)):
            if count >= max_per_config:
                break

            # ── Geographic filter (cheap, no decode) ────────────────────────
            lat = ex.get("lat") or ex.get("latitude")
            lon = ex.get("long") or ex.get("longitude")
            if not skip_geo and not _in_bbox(lat, lon, bbox):
                continue

            # ── XC ID dedup against train.csv ────────────────────────────────
            filepath = str(ex.get("filepath", ""))
            xc_id = _extract_xc_id(filepath)
            if xc_id and xc_id in train_xc_ids:
                continue

            # ── Quality filter: require detected_events ──────────────────────
            det_events = ex.get("detected_events") or []
            if require_detected_events and not det_events:
                continue

            # ── Source ID dedup against manifest ────────────────────────────
            source_id = f"birdset:{config_name}:{split}:{filepath or count}"
            if skip_existing and source_id in dedup.source_ids:
                continue

            # ── Audio decode ─────────────────────────────────────────────────
            audio_field = ex.get("audio")
            if not isinstance(audio_field, dict):
                continue
            raw_bytes = audio_field.get("bytes")
            if not raw_bytes:
                continue

            try:
                arr, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
            except Exception as e:
                logger.debug(f"soundfile decode failed ({config_name} ex {count}): {e}")
                continue
            if arr.ndim > 1:
                arr = arr.mean(axis=1)

            if sr != target_sr:
                try:
                    import resampy
                    arr = resampy.resample(arr, sr, target_sr)
                except ImportError:
                    from scipy.signal import resample_poly
                    import math
                    g = math.gcd(target_sr, sr)
                    arr = resample_poly(arr, target_sr // g, sr // g).astype("float32")
                sr = target_sr

            # ── Extract 5s window around first detected_events timestamp ─────
            # detected_events entries look like {"start_time": 1.2, "end_time": 3.7, ...}
            # or may just be floats. Fall back to start of clip if no timestamps.
            event_start = 0.0
            if det_events:
                first = det_events[0]
                if isinstance(first, dict):
                    event_start = float(first.get("start_time") or first.get("start") or 0.0)
                elif isinstance(first, (int, float)):
                    event_start = float(first)

            # Center the 5s window on the event, clamped to clip boundaries
            clip_dur = len(arr) / sr
            win_start = max(0.0, min(event_start - seg_dur / 2, clip_dur - seg_dur))
            win_start = max(0.0, win_start)
            s0 = int(win_start * sr)
            s1 = s0 + int(seg_dur * sr)
            segment = arr[s0:s1]

            # Zero-pad if clip is shorter than 5s
            target_len = int(seg_dur * sr)
            if len(segment) < target_len:
                import numpy as _np
                segment = _np.pad(segment, (0, target_len - len(segment)))

            # ── Resolve ebird code ───────────────────────────────────────────
            raw_ebird = ex.get("ebird_code")
            if isinstance(raw_ebird, int) and int2ebird:
                ebird_code = int2ebird.get(raw_ebird, "")
            else:
                ebird_code = str(raw_ebird or "").lower()

            tax = ebird_taxonomy.get(ebird_code, {})

            # ── Write 5s OGG segment (compressed, ~150KB vs ~5MB WAV) ────────
            basename = Path(filepath).stem if filepath else f"sample_{count}"
            out_name = f"birdset_{config_name}_{split}_{basename}.ogg"
            out_path = output_base / config_name / split / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if skip_existing and out_path.name in dedup.basenames:
                continue

            sf.write(str(out_path), segment, sr, format="OGG", subtype="VORBIS")
            sha1 = _sha1_path(out_path)

            if skip_existing and sha1 in dedup.hashes:
                out_path.unlink(missing_ok=True)
                continue

            rel_out = out_path.relative_to(output_dir)
            row = {
                "source": "birdset",
                "source_split": split,
                "source_id": source_id,
                "species_name": tax.get("scientific_name") or ebird_code,
                "common_name": tax.get("common_name", ""),
                "supercategory": "aves",
                "latitude": _safe_float(lat),
                "longitude": _safe_float(lon),
                "duration_sec": seg_dur,
                "sample_rate": sr,
                "relative_path": str(rel_out),
                "sha1": sha1,
            }
            new_rows.append(row)
            dedup.source_ids.add(source_id)
            dedup.hashes.add(sha1)
            dedup.basenames.add(out_path.name)
            count += 1

        append_manifest_rows(manifest_path, new_rows)
        total_written += count
        logger.info(f"BirdSet {config_name}: {count} clips written")

    logger.info(f"BirdSet geo-streaming complete: {total_written} total clips across {len(configs)} configs")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Filtered distillation data collector (iNat + BirdSet)")
    p.add_argument("--output-dir", type=str, default="data/distill_audio")
    p.add_argument("--manifest-path", type=str, default="data/distill_manifest.csv")
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--skip-existing", action="store_true")

    p.add_argument("--inat", action="store_true", help="Enable iNat Sounds collection")
    p.add_argument("--inat-max-files", type=int, default=5000)
    p.add_argument("--inat-splits", type=str, default="train", help="Comma list: train,val,test")
    p.add_argument(
        "--inat-supercategories",
        type=str,
        default="aves,amphibia,insecta,mammalia,reptilia",
        help="Comma list of supercategories to keep",
    )
    p.add_argument(
        "--bbox",
        type=str,
        default="",
        help="min_lat,min_lon,max_lat,max_lon (use --bbox=-25,-60,-10,-45 for negative coords)",
    )
    p.add_argument("--inat-min-duration", type=float, default=2.0)

    p.add_argument("--birdset", action="store_true", help="Enable BirdSet collection (legacy single-config, non-streaming)")
    p.add_argument(
        "--birdset-configs",
        type=str,
        default="PER",
        help="Comma-separated BirdSet config names (used by both --birdset and --birdset-geo)",
    )
    p.add_argument("--birdset-split", type=str, default="train")
    p.add_argument("--birdset-max-files", type=int, default=3000, help="Max files per config (legacy --birdset)")
    p.add_argument("--birdset-species-allowlist", type=str, default="")

    # Geographic streaming mode (preferred)
    p.add_argument(
        "--birdset-geo",
        action="store_true",
        help="Enable multi-config geographic streaming (all configs in --birdset-configs, bbox filtered)",
    )
    p.add_argument(
        "--birdset-geo-max-per-config",
        type=int,
        default=5000,
        help="Max clips per BirdSet config in geo-streaming mode",
    )
    p.add_argument(
        "--birdset-geo-bbox",
        type=str,
        default="-22,-62,-10,-44",
        help="Pantanal bbox: min_lat,min_lon,max_lat,max_lon (default: Pantanal region)",
    )
    p.add_argument(
        "--birdset-require-detected-events",
        action="store_true",
        default=True,
        help="In geo mode, only keep clips that have detected_events annotations",
    )
    p.add_argument(
        "--birdset-no-bbox-configs",
        type=str,
        default="PER,NES",
        help="Comma-separated configs that skip the bbox filter (region-specific datasets "
             "where every clip is relevant). Default: PER,NES",
    )
    p.add_argument(
        "--train-csv",
        type=str,
        default="data/train.csv",
        help="Path to competition train.csv for pre-embed XC ID dedup",
    )

    p.add_argument(
        "--taxonomy-cache-dir",
        type=str,
        default="data",
        help="Directory containing or to cache eBird taxonomy CSV (searches for eBird_taxonomy*.csv)",
    )
    p.add_argument(
        "--repair-manifest",
        action="store_true",
        help="Retroactively fix integer ebird_code labels in existing BirdSet manifest rows",
    )
    return p


def repair_manifest_birdset_labels(
    manifest_path: Path,
    ebird_taxonomy: dict[str, dict[str, str]],
    logger: logging.Logger,
):
    """Fix existing BirdSet manifest rows where species_name is an integer label index.

    Builds int→ebird_code maps per config from the ClassLabel feature, then rewrites
    species_name/common_name for any row where species_name looks like a raw integer.
    """
    if not manifest_path.exists():
        logger.warning("Manifest not found, nothing to repair.")
        return

    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Collect which BirdSet configs need a label map
    configs_needed: set[str] = set()
    for row in rows:
        if row.get("source") != "birdset":
            continue
        val = row.get("species_name", "").strip()
        if val.lstrip("-").isdigit():
            # Extract config from source_id: "birdset:PER:train:..."
            parts = row.get("source_id", "").split(":")
            if len(parts) >= 2:
                configs_needed.add(parts[1])

    if not configs_needed:
        logger.info("No integer BirdSet labels found — manifest is already clean.")
        return

    logger.info(f"Repairing BirdSet labels for configs: {configs_needed}")
    int2ebird_by_config: dict[str, dict[int, str]] = {}
    for cfg in configs_needed:
        int2ebird_by_config[cfg] = build_birdset_label_map(cfg, "train", logger)

    repaired = 0
    for row in rows:
        if row.get("source") != "birdset":
            continue
        val = row.get("species_name", "").strip()
        if not val.lstrip("-").isdigit():
            continue

        parts = row.get("source_id", "").split(":")
        cfg = parts[1] if len(parts) >= 2 else ""
        int2ebird = int2ebird_by_config.get(cfg, {})
        ebird_code = int2ebird.get(int(val), "")
        if not ebird_code:
            continue

        tax = ebird_taxonomy.get(ebird_code.lower(), {})
        row["species_name"] = tax.get("scientific_name") or ebird_code
        row["common_name"] = tax.get("common_name", "")
        repaired += 1

    # Rewrite manifest in place
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Repaired {repaired} BirdSet rows in {manifest_path}")


def main():
    args = build_parser().parse_args()
    logger = setup_logger(args.log_level)

    output_dir = Path(args.output_dir).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    taxonomy_cache_dir = Path(args.taxonomy_cache_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_manifest(manifest_path)
    dedup = load_dedup_index(manifest_path, output_dir)
    logger.info(
        f"Loaded dedup index: source_ids={len(dedup.source_ids)}, "
        f"hashes={len(dedup.hashes)}, basenames={len(dedup.basenames)}"
    )

    if not args.inat and not args.birdset and not args.birdset_geo and not args.repair_manifest:
        raise ValueError("Nothing to do: set at least one of --inat, --birdset, --birdset-geo, or --repair-manifest")

    # Load eBird taxonomy once if BirdSet or repair is requested
    ebird_taxonomy: dict[str, dict[str, str]] = {}
    if args.birdset or args.birdset_geo or args.repair_manifest:
        ebird_taxonomy = load_ebird_taxonomy(taxonomy_cache_dir, logger)

    if args.repair_manifest:
        repair_manifest_birdset_labels(manifest_path, ebird_taxonomy, logger)

    if args.inat:
        inat_splits = [s.strip() for s in args.inat_splits.split(",") if s.strip()]
        invalid = [s for s in inat_splits if s not in INAT_AUDIO_URLS]
        if invalid:
            raise ValueError(f"Invalid --inat-splits values: {invalid}")

        supercats = {s.strip().lower() for s in args.inat_supercategories.split(",") if s.strip()}
        bbox = _parse_bbox(args.bbox) if args.bbox else None

        collect_inat(
            output_dir=output_dir,
            manifest_path=manifest_path,
            dedup=dedup,
            supercategories=supercats,
            max_files=args.inat_max_files,
            bbox=bbox,
            min_duration=args.inat_min_duration,
            include_splits=inat_splits,
            skip_existing=args.skip_existing,
            logger=logger,
        )

    if args.birdset:
        allowlist_path = Path(args.birdset_species_allowlist).resolve() if args.birdset_species_allowlist else None
        birdset_configs = [c.strip() for c in args.birdset_configs.split(",") if c.strip()]
        for config_name in birdset_configs:
            collect_birdset(
                output_dir=output_dir,
                manifest_path=manifest_path,
                dedup=dedup,
                config_name=config_name,
                split=args.birdset_split,
                max_files=args.birdset_max_files,
                species_allowlist_path=allowlist_path,
                ebird_taxonomy=ebird_taxonomy,
                skip_existing=args.skip_existing,
                logger=logger,
            )

    if args.birdset_geo:
        birdset_configs = [c.strip() for c in args.birdset_configs.split(",") if c.strip()]
        geo_bbox = _parse_bbox(args.birdset_geo_bbox)
        train_csv_path = Path(args.train_csv).resolve() if args.train_csv else None
        train_xc_ids = load_train_xc_ids(train_csv_path, logger)

        no_bbox_configs = {
            c.strip() for c in args.birdset_no_bbox_configs.split(",") if c.strip()
        }
        collect_birdset_streaming_geo(
            output_dir=output_dir,
            manifest_path=manifest_path,
            dedup=dedup,
            configs=birdset_configs,
            split=args.birdset_split,
            bbox=geo_bbox,
            max_per_config=args.birdset_geo_max_per_config,
            require_detected_events=args.birdset_require_detected_events,
            ebird_taxonomy=ebird_taxonomy,
            train_xc_ids=train_xc_ids,
            skip_existing=args.skip_existing,
            logger=logger,
            no_bbox_configs=no_bbox_configs,
        )

    logger.info(f"Done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
