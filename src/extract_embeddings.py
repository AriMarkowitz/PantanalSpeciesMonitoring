"""Stage 1: Extract Perch 2.0 embeddings for all segments.

Reads segments.csv from Stage 0, loads audio on-the-fly, runs through
Perch 2.0, and writes embeddings to HDF5.

Supports resumability: skips segments already present in the output HDF5.

Usage:
    python src/extract_embeddings.py
    python src/extract_embeddings.py --set stage1.batch_size=128
"""

import os
import sys
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from tqdm import tqdm

from config import get_config
from utils import load_audio_segment, setup_logging


def init_perch(model_name: str, logger):
    """Load Perch model. Imports tensorflow only when needed."""
    logger.info(f"Loading Perch model: {model_name}")
    from perch_hoplite.zoo import model_configs
    model = model_configs.load_model_by_name(model_name)
    logger.info("Perch model loaded successfully")
    return model


def embed_batch(model, waveforms: list[np.ndarray], logger):
    """Run Perch on a batch of waveforms.

    Returns dicts with 'embedding', 'spatial', 'logits' arrays.
    Perch's embed() is per-sample, so we loop (model is already on GPU).
    """
    results = []
    for wav in waveforms:
        try:
            outputs = model.embed(wav)
            results.append({
                "embedding": np.asarray(outputs.embeddings, dtype=np.float32),
                "logits": outputs.logits,
            })
        except Exception as e:
            logger.warning(f"Embed failed for waveform shape {wav.shape}: {e}")
            results.append(None)
    return results


def extract_top_k_logits(logits_dict, k: int):
    """Extract top-K logits from Perch output.

    Perch returns logits as a dict {'label': array} or a flat array.
    Handle both formats.
    """
    if isinstance(logits_dict, dict):
        # Perch v2 returns {'label': np.array of shape (n_classes,)}
        if "label" in logits_dict:
            logits = np.asarray(logits_dict["label"], dtype=np.float32).ravel()
        else:
            # Concatenate all logit arrays
            logits = np.concatenate(
                [np.asarray(v, dtype=np.float32).ravel()
                 for v in logits_dict.values()]
            )
    else:
        logits = np.asarray(logits_dict, dtype=np.float32).ravel()

    if len(logits) <= k:
        indices = np.arange(len(logits), dtype=np.int16)
        values = logits.astype(np.float16)
    else:
        top_idx = np.argpartition(logits, -k)[-k:]
        top_idx = top_idx[np.argsort(logits[top_idx])[::-1]]
        indices = top_idx.astype(np.int16)
        values = logits[top_idx].astype(np.float16)

    return values, indices


def create_h5(path: str, n_segments: int, cfg: dict):
    """Create HDF5 file with pre-allocated datasets."""
    k = cfg["stage1"]["top_k_logits"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.create_dataset(
            "global_embeddings",
            shape=(n_segments, 1536),
            dtype="float32",
            chunks=(min(1000, n_segments), 1536),
            compression="lzf",
        )
        if cfg["stage1"]["store_spatial"]:
            f.create_dataset(
                "spatial_embeddings",
                shape=(n_segments, 5, 3, 1536),
                dtype="float32",
                chunks=(min(100, n_segments), 5, 3, 1536),
                compression="lzf",
            )
        if cfg["stage1"]["store_logits"]:
            f.create_dataset(
                "logit_values",
                shape=(n_segments, k),
                dtype="float16",
                chunks=(min(1000, n_segments), k),
            )
            f.create_dataset(
                "logit_indices",
                shape=(n_segments, k),
                dtype="int16",
                chunks=(min(1000, n_segments), k),
            )
        # Track which rows are written (for resumability)
        f.create_dataset(
            "written",
            shape=(n_segments,),
            dtype="bool",
            fillvalue=False,
        )
        # Store segment IDs as variable-length strings
        dt = h5py.string_dtype()
        f.create_dataset("segment_ids", shape=(n_segments,), dtype=dt)


def main(cfg: dict):
    logger = setup_logging("extract_embeddings", cfg["outputs"]["logs_dir"])
    logger.info("Stage 1: Embedding extraction")

    # Load segment metadata
    segments_csv = cfg["outputs"]["segments_csv"]
    segments = pd.read_csv(segments_csv, low_memory=False)
    n = len(segments)
    logger.info(f"Loaded {n} segments from {segments_csv}")

    sr = cfg["data"]["sample_rate"]
    seg_dur = cfg["data"]["segment_duration"]
    h5_path = cfg["outputs"]["embeddings_h5"]
    batch_size = cfg["stage1"]["batch_size"]
    top_k = cfg["stage1"]["top_k_logits"]

    # Create or open HDF5
    if not os.path.exists(h5_path):
        logger.info(f"Creating HDF5: {h5_path}")
        create_h5(h5_path, n, cfg)
    else:
        logger.info(f"Resuming into existing HDF5: {h5_path}")

    # Check what's already done
    with h5py.File(h5_path, "r") as f:
        written = f["written"][:]
        already_done = int(written.sum())
    if already_done > 0:
        logger.info(f"Resuming: {already_done}/{n} segments already embedded")
    if already_done == n:
        logger.info("All segments already embedded, nothing to do")
        return

    # Load Perch
    model = init_perch(cfg["stage1"]["model_name"], logger)

    # Process in batches
    todo_indices = np.where(~written)[0]
    logger.info(f"Embedding {len(todo_indices)} remaining segments")

    h5f = h5py.File(h5_path, "r+")
    try:
        for batch_start in tqdm(range(0, len(todo_indices), batch_size),
                                desc="Embedding"):
            batch_idx = todo_indices[batch_start:batch_start + batch_size]

            # Load audio
            waveforms = []
            valid_idx = []
            for i in batch_idx:
                row = segments.iloc[i]
                try:
                    wav = load_audio_segment(
                        row["source_file"], row["start_sec"],
                        row["end_sec"], sr
                    )
                    # Pad or trim to exact segment length
                    expected_len = int(seg_dur * sr)
                    if len(wav) < expected_len:
                        wav = np.pad(wav, (0, expected_len - len(wav)))
                    elif len(wav) > expected_len:
                        wav = wav[:expected_len]
                    waveforms.append(wav)
                    valid_idx.append(i)
                except Exception as e:
                    logger.warning(f"Failed to load segment {i} "
                                   f"({row['segment_id']}): {e}")

            if not waveforms:
                continue

            # Run Perch
            results = embed_batch(model, waveforms, logger)

            # Write to HDF5
            for idx, result in zip(valid_idx, results):
                if result is None:
                    continue

                emb = result["embedding"]

                # Handle global vs spatial embedding shapes
                if emb.ndim == 3 and emb.shape == (5, 3, 1536):
                    # Perch returned spatial; derive global by mean-pooling
                    if cfg["stage1"]["store_spatial"]:
                        h5f["spatial_embeddings"][idx] = emb
                    global_emb = emb.mean(axis=(0, 1))
                    h5f["global_embeddings"][idx] = global_emb
                elif emb.ndim == 1 and emb.shape[0] == 1536:
                    h5f["global_embeddings"][idx] = emb
                else:
                    # Try to handle other shapes
                    flat = emb.ravel()
                    if len(flat) == 1536:
                        h5f["global_embeddings"][idx] = flat
                    else:
                        logger.warning(f"Unexpected embedding shape {emb.shape} "
                                       f"for segment {idx}")
                        continue

                # Logits
                if cfg["stage1"]["store_logits"] and result["logits"] is not None:
                    values, indices = extract_top_k_logits(
                        result["logits"], top_k
                    )
                    # Pad if fewer than top_k logits
                    if len(values) < top_k:
                        values = np.pad(values, (0, top_k - len(values)))
                        indices = np.pad(indices, (0, top_k - len(indices)),
                                         constant_values=-1)
                    h5f["logit_values"][idx] = values
                    h5f["logit_indices"][idx] = indices

                h5f["segment_ids"][idx] = segments.iloc[idx]["segment_id"]
                h5f["written"][idx] = True

            # Flush periodically
            if (batch_start // batch_size) % 10 == 0:
                h5f.flush()

    except KeyboardInterrupt:
        logger.info("Interrupted — flushing HDF5 before exit")
    finally:
        h5f.flush()
        h5f.close()

    # Final stats
    with h5py.File(h5_path, "r") as f:
        done = int(f["written"][:].sum())
    logger.info(f"Embedding complete: {done}/{n} segments written to {h5_path}")


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
