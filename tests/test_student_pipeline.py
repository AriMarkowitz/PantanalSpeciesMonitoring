"""Smoke test for the student distillation data pipeline.

Verifies: mel caching, memmap loading, dataset indexing, DataLoader
with multiple workers, and tensor shapes.

Usage:
    python tests/test_student_pipeline.py
"""
import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_config
from cache_mels import wav_to_mel_np, get_mel_shape, cache_mels_memmap
from train_student import DistillDataset, collate_fn


def main():
    cfg = get_config()
    h5_path = cfg["outputs"]["embeddings_h5"]
    csv_path = cfg["outputs"]["segments_csv"]

    # 1. Test mel computation
    print("=== Mel computation ===")
    sr = cfg["data"]["sample_rate"]
    mel_cfg = cfg.get("student_mel", {})
    wav = np.random.randn(sr * 5).astype(np.float32) * 0.1
    t0 = time.time()
    mel = wav_to_mel_np(wav, sr, mel_cfg)
    print(f"  Shape: {mel.shape}, dtype: {mel.dtype}")
    print(f"  min={mel.min():.4f}, max={mel.max():.4f}, mean={mel.mean():.4f}")
    n_mels, T = get_mel_shape(cfg)
    assert mel.shape == (n_mels, T), f"Bad shape: {mel.shape} vs ({n_mels}, {T})"
    print("  PASS")

    # 2. Check memmap exists
    memmap_path = h5_path + ".mels.npy"
    print(f"\n=== Memmap check: {memmap_path} ===")
    if not os.path.exists(memmap_path):
        print("  Not found — need to run cache first (or test with a small subset)")
        return

    N_total = 358454  # approximate
    mmap = np.memmap(memmap_path, dtype=np.float16, mode="r",
                     shape=(N_total, n_mels, T))
    # Check a few samples
    import h5py
    with h5py.File(h5_path, "r") as h5:
        written = h5["written"][:]
    valid = np.where(written)[0][:5]
    for i in valid:
        m = mmap[i]
        print(f"  [{i}] min={m.min():.3f} max={m.max():.3f} mean={m.mean():.3f}")
        assert m.max() > 0.01, f"Sample {i} appears all-zero"
    print("  PASS")

    # 3. Test dataset
    print("\n=== Dataset loading ===")
    t0 = time.time()
    ds = DistillDataset(csv_path, h5_path, cfg, fold=0, split="train")
    print(f"  Load time: {time.time()-t0:.1f}s, N={len(ds)}")

    # 4. Single item
    print("\n=== Single __getitem__ ===")
    t0 = time.time()
    mel, g_emb, s_emb, logits = ds[0]
    print(f"  mel: {mel.shape}, min={mel.min():.3f}, max={mel.max():.3f}, "
          f"mean={mel.mean():.3f}, time={((time.time()-t0)*1000):.1f}ms")
    assert mel.shape == (1, n_mels, T)
    assert mel.mean() > 0.01, f"Mel all zeros: mean={mel.mean()}"
    print("  PASS")

    # 5. Benchmark
    print("\n=== Benchmark: 1000 items ===")
    t0 = time.time()
    for j in range(1000):
        ds[j]
    elapsed = time.time() - t0
    print(f"  1000 items in {elapsed*1000:.0f}ms ({elapsed/1000*1000:.2f}ms/item)")

    # 6. DataLoader
    print("\n=== DataLoader (num_workers=4, 5 batches) ===")
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=4,
                    collate_fn=collate_fn, pin_memory=True)
    t0 = time.time()
    for batch_idx, (mels, g, s, l) in enumerate(dl):
        elapsed = time.time() - t0
        print(f"  Batch {batch_idx}: mels={mels.shape}, "
              f"mean={mels.mean():.3f}, time={elapsed:.2f}s")
        if batch_idx >= 4:
            break
    print("  PASS")

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
