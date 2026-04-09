"""Quick smoke test for the student distillation data pipeline.

Verifies: HDF5 loading into RAM, mel spectrogram computation, dataset
indexing, DataLoader with multiple workers, and tensor shapes.

Usage:
    python tests/test_student_pipeline.py
"""
import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_config
from train_student import DistillDataset, wav_to_mel, collate_fn


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
    mel = wav_to_mel(wav, sr, mel_cfg)
    print(f"  Shape: {mel.shape}, dtype: {mel.dtype}")
    print(f"  min={mel.min():.4f}, max={mel.max():.4f}, mean={mel.mean():.4f}")
    print(f"  Time: {(time.time()-t0)*1000:.1f}ms")
    assert mel.shape == (1, 128, 501), f"Bad shape: {mel.shape}"
    assert mel.min() >= 0 and mel.max() <= 1.01, "Values out of [0,1]"
    print("  PASS")

    # 2. Test dataset loading
    print("\n=== Dataset loading ===")
    t0 = time.time()
    ds = DistillDataset(csv_path, h5_path, cfg, fold=0, split="train")
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.1f}s")
    print(f"  N samples: {len(ds)}")
    print(f"  Global embs shape: {ds.global_embs.shape}")
    if ds.has_spatial:
        print(f"  Spatial embs shape: {ds.spatial_embs.shape}")
    mem_gb = (ds.global_embs.nbytes +
              (ds.spatial_embs.nbytes if ds.has_spatial else 0)) / 1e9
    print(f"  RAM usage: {mem_gb:.1f}GB")

    # 3. Test single item
    print("\n=== Single __getitem__ ===")
    t0 = time.time()
    mel, g_emb, s_emb, logits = ds[0]
    item_time = time.time() - t0
    print(f"  mel: {mel.shape}, dtype={mel.dtype}, "
          f"min={mel.min():.3f}, max={mel.max():.3f}, mean={mel.mean():.3f}")
    print(f"  global_emb: {g_emb.shape}")
    print(f"  spatial_emb: {s_emb.shape}")
    print(f"  logits: {logits.shape}")
    print(f"  Time: {item_time*1000:.1f}ms")
    assert mel.shape == (1, 128, 501), f"Bad mel shape: {mel.shape}"
    assert g_emb.shape == (1536,), f"Bad global shape: {g_emb.shape}"
    assert s_emb.shape == (5, 1536), f"Bad spatial shape: {s_emb.shape}"
    assert mel.mean() > 0.01, f"Mel appears to be all zeros (mean={mel.mean():.6f})"
    print("  PASS")

    # 4. Test DataLoader with workers
    print("\n=== DataLoader (num_workers=4, 3 batches) ===")
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4,
                    collate_fn=collate_fn, pin_memory=False)
    t0 = time.time()
    for batch_idx, (mels, g, s, l) in enumerate(dl):
        elapsed = time.time() - t0
        print(f"  Batch {batch_idx}: mels={mels.shape}, "
              f"mean={mels.mean():.3f}, time={elapsed:.1f}s")
        if batch_idx >= 2:
            break
    print("  PASS — DataLoader works with multiple workers")

    # 5. Quick benchmark: 10 items
    print("\n=== Benchmark: 10 items sequential ===")
    t0 = time.time()
    for i in range(10):
        ds[i]
    print(f"  10 items in {(time.time()-t0)*1000:.0f}ms "
          f"({(time.time()-t0)/10*1000:.0f}ms/item)")

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
