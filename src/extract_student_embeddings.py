"""Extract student embeddings for all training segments.

Runs the trained student model on all segments (using cached mels) and
writes global + spatial embeddings to a new HDF5 file. These embeddings
replace the Perch teacher embeddings for downstream feature building
and classifier training, ensuring no train/inference distribution mismatch.

Usage:
    python src/extract_student_embeddings.py --ckpt outputs/checkpoints/student/best.pt
    python src/extract_student_embeddings.py  # auto-selects best checkpoint
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import h5py
import torch
from pathlib import Path
from tqdm import tqdm

from config import get_config
from student_model import StudentEmbedder
from cache_mels import get_mel_shape
from utils import setup_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="",
                        help="Student checkpoint path (auto-selects latest if empty)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    args, _ = parser.parse_known_args()

    cfg = get_config()
    logger = setup_logging("extract_student_emb", cfg["outputs"]["logs_dir"])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Find best student checkpoint
    ckpt_path = args.ckpt
    if not ckpt_path:
        ckpt_dir = Path(cfg["outputs"]["checkpoints_dir"]) / "student"
        candidates = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError("No student checkpoint found")
        ckpt_path = str(candidates[-1])
    logger.info(f"Loading student from: {ckpt_path}")

    # Load student
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    student = StudentEmbedder.from_pretrained()
    student.load_state_dict(state["model_state_dict"])
    student = student.to(device).eval()
    logger.info(f"Student loaded (val_cos_sim={state.get('val_cosine_sim', '?')})")

    # Load cached mels (memmap) — supports MELS_CACHE_DIR override for HPC scratch
    from cache_mels import _resolve_mels_path
    h5_path = cfg["outputs"]["embeddings_h5"]
    memmap_path = _resolve_mels_path(h5_path)
    if not Path(memmap_path).exists():
        raise FileNotFoundError(f"Mel cache not found: {memmap_path}")

    with h5py.File(h5_path, "r") as h5:
        written = h5["written"][:]
    N = len(written)
    n_mels, T = get_mel_shape(cfg)
    mels = np.memmap(memmap_path, dtype=np.float16, mode="r", shape=(N, n_mels, T))
    valid_indices = np.where(written)[0]
    logger.info(f"Loaded mel cache: {N} total, {len(valid_indices)} valid")

    # Output HDF5 — student embeddings alongside the original
    out_path = str(Path(h5_path).parent / "student_embeddings.h5")
    logger.info(f"Output: {out_path}")

    embed_dim = cfg["stage1d"]["embed_dim"]  # 1536
    spatial_h = 5

    with h5py.File(out_path, "w") as out:
        out.create_dataset("global_embeddings", shape=(N, embed_dim),
                           dtype="float32", chunks=(64, embed_dim))
        out.create_dataset("spatial_embeddings", shape=(N, spatial_h, embed_dim),
                           dtype="float32", chunks=(64, spatial_h, embed_dim))
        out.create_dataset("written", data=written)

        # Copy over metadata from original HDF5
        with h5py.File(h5_path, "r") as orig:
            if "segment_ids" in orig:
                out.create_dataset("segment_ids", data=orig["segment_ids"][:])
            # Copy logits as-is (Perch logits, still useful for features)
            if "logit_values" in orig:
                out.create_dataset("logit_values", data=orig["logit_values"][:])
            if "logit_indices" in orig:
                out.create_dataset("logit_indices", data=orig["logit_indices"][:])

        # Extract in batches
        bs = args.batch_size
        n_done = 0

        for start in tqdm(range(0, len(valid_indices), bs),
                          desc="Extracting", disable=not True):
            batch_idx = valid_indices[start:start + bs]

            # Load mels from memmap → tensor
            mel_batch = torch.from_numpy(
                mels[batch_idx].astype(np.float32)
            ).unsqueeze(1).to(device)  # (B, 1, n_mels, T)

            with torch.no_grad():
                g_emb, s_emb = student(mel_batch, normalize=True)

            g_np = g_emb.cpu().numpy()  # (B, 1536)
            s_np = s_emb.cpu().numpy()  # (B, 5, 1536)

            out["global_embeddings"][batch_idx] = g_np
            out["spatial_embeddings"][batch_idx] = s_np

            n_done += len(batch_idx)

        logger.info(f"Extracted {n_done} student embeddings → {out_path}")

    # Also extract distill corpus if it exists
    distill_h5 = cfg["outputs"].get("distill_embeddings_h5", "")
    distill_memmap = _resolve_mels_path(distill_h5) if distill_h5 else ""
    if distill_h5 and Path(distill_memmap).exists():
        logger.info(f"Extracting distill corpus...")
        with h5py.File(distill_h5, "r") as h5d:
            written_d = h5d["written"][:]
        Nd = len(written_d)
        mels_d = np.memmap(distill_memmap, dtype=np.float16, mode="r",
                           shape=(Nd, n_mels, T))
        valid_d = np.where(written_d)[0]

        out_d_path = str(Path(distill_h5).parent / "student_distill_embeddings.h5")
        with h5py.File(out_d_path, "w") as out_d:
            out_d.create_dataset("global_embeddings", shape=(Nd, embed_dim),
                                 dtype="float32", chunks=(64, embed_dim))
            out_d.create_dataset("spatial_embeddings", shape=(Nd, spatial_h, embed_dim),
                                 dtype="float32", chunks=(64, spatial_h, embed_dim))
            out_d.create_dataset("written", data=written_d)

            for start in tqdm(range(0, len(valid_d), bs), desc="Distill"):
                batch_idx = valid_d[start:start + bs]
                mel_batch = torch.from_numpy(
                    mels_d[batch_idx].astype(np.float32)
                ).unsqueeze(1).to(device)

                with torch.no_grad():
                    g_emb, s_emb = student(mel_batch, normalize=True)

                out_d["global_embeddings"][batch_idx] = g_emb.cpu().numpy()
                out_d["spatial_embeddings"][batch_idx] = s_emb.cpu().numpy()

        logger.info(f"Extracted {len(valid_d)} distill embeddings → {out_d_path}")

    logger.info("Done")


if __name__ == "__main__":
    main()
