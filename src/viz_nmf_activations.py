"""Visualize per-species pseudoinverse H activations next to mel spectrograms.

Loads the learned per-species dictionaries, picks a strong_primary clip for
a chosen species, and renders:

    row 0: mel spectrogram V
    row 1: H_sp for the true species         (k_sp, T)
    row 2: H_sp for the species with the highest activation energy (excl. true)
    row 3: H_sp for the species with the 2nd-highest activation energy

H is the UNCONSTRAINED pseudoinverse projection H = W_all_pinv @ V, sliced
per species. Activation energy is ||max(H_sp, 0)||_F^2 — the non-negative
(NMF-valid) part of the reconstruction contribution. Negative entries remain
visible in the heatmap.

Usage:
    python src/viz_nmf_activations.py                         # random strong_primary clip
    python src/viz_nmf_activations.py --species baymac        # specific target species
    python src/viz_nmf_activations.py --species baymac --segment-id ...
    python src/viz_nmf_activations.py --out outputs/nmf_viz/  # save all three panels
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

from config import get_config
from cache_mels import get_mel_shape
from utils import load_audio_segment


def _compute_species_recon_err(
    V: np.ndarray,
    W_all: np.ndarray,
    W_pinv: np.ndarray,
    boundaries: np.ndarray,
) -> np.ndarray:
    """Per-species relative reconstruction error for one clip.

    Returns (num_species,) array of ||V - W_sp H_sp||^2 / ||V||^2.
    """
    H_all = W_pinv @ V                              # (sum_k, T)
    V_sq = float(np.sum(V ** 2)) + 1e-10
    num_species = len(boundaries) - 1
    errs = np.zeros(num_species, dtype=np.float32)
    for sp_idx in range(num_species):
        c0, c1 = int(boundaries[sp_idx]), int(boundaries[sp_idx + 1])
        if c1 <= c0:
            errs[sp_idx] = np.nan
            continue
        W_sp = W_all[:, c0:c1]
        H_sp = H_all[c0:c1, :]
        V_hat = W_sp @ H_sp
        errs[sp_idx] = float(np.sum((V - V_hat) ** 2)) / V_sq
    return errs


def _slice_H(H_all: np.ndarray, boundaries: np.ndarray, sp_idx: int) -> np.ndarray:
    c0, c1 = int(boundaries[sp_idx]), int(boundaries[sp_idx + 1])
    return H_all[c0:c1, :]


def _figure_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


def _wav_to_wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    """Encode a float32 waveform as an in-memory WAV file."""
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _write_html_page(png_bytes: bytes, wav_bytes: bytes, title: str,
                      duration_sec: float, T_frames: int,
                      out_path: Path) -> None:
    """Write a self-contained HTML page: audio player + figure + time cursor.

    The time cursor is a semi-transparent vertical line over the figure,
    driven by audio.currentTime. Works offline in any browser.
    """
    png_b64 = base64.b64encode(png_bytes).decode("ascii")
    wav_b64 = base64.b64encode(wav_bytes).decode("ascii")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  body {{ font-family: sans-serif; margin: 20px; background: #fafafa; }}
  .wrap {{ position: relative; display: inline-block; }}
  img {{ display: block; max-width: 100%; }}
  #cursor {{
    position: absolute; top: 0; bottom: 0; width: 2px;
    background: rgba(0,255,0,0.75); pointer-events: none;
    left: 0;
  }}
  .meta {{ color: #555; margin-bottom: 10px; font-size: 14px; }}
</style></head>
<body>
  <h2>{title}</h2>
  <div class="meta">Duration: {duration_sec:.2f}s &nbsp;|&nbsp; {T_frames} frames</div>
  <audio id="player" controls src="data:audio/wav;base64,{wav_b64}"></audio>
  <div class="wrap" id="figwrap">
    <img id="fig" src="data:image/png;base64,{png_b64}">
    <div id="cursor"></div>
  </div>
<script>
  // Plot area starts/ends within the PNG (matplotlib leaves margins).
  // We measure the displayed image and guess the plot region; if the
  // cursor drifts, tweak LEFT_FRAC / RIGHT_FRAC below.
  const LEFT_FRAC  = 0.09;   // left margin of plot as fraction of image width
  const RIGHT_FRAC = 0.93;   // right edge of plot (before colorbar)
  const DURATION  = {duration_sec};
  const audio = document.getElementById('player');
  const cursor = document.getElementById('cursor');
  const fig = document.getElementById('fig');

  function update() {{
    const w = fig.clientWidth;
    const frac = Math.max(0, Math.min(1, audio.currentTime / DURATION));
    const x = w * (LEFT_FRAC + frac * (RIGHT_FRAC - LEFT_FRAC));
    cursor.style.left = x + 'px';
    if (!audio.paused) requestAnimationFrame(update);
  }}
  audio.addEventListener('play', update);
  audio.addEventListener('timeupdate', update);
  audio.addEventListener('seeked', update);
  fig.addEventListener('load', update);
</script>
</body></html>
"""
    out_path.write_text(html)


def visualize_one_clip(
    cfg: dict,
    species: str | None,
    segment_id: str | None,
    out_path: str | None,
    seed: int,
    html: bool = False,
):
    # ── Load artifacts ───────────────────────────────────────────────────────
    nmf_dir = Path(cfg["outputs"]["nmf_dir"])
    W_all = np.load(nmf_dir / "W_all.npy")
    W_pinv = np.load(nmf_dir / "W_all_pinv.npy")
    boundaries = np.load(nmf_dir / "species_boundaries.npy")
    with open(nmf_dir / "species_order.json") as f:
        species_order = json.load(f)
    sp_to_idx = {s: i for i, s in enumerate(species_order)}

    # ── Pick a clip ──────────────────────────────────────────────────────────
    segments = pd.read_csv(cfg["outputs"]["segments_csv"], low_memory=False)
    emb_h5 = cfg["outputs"]["embeddings_h5"]
    with h5py.File(emb_h5, "r") as h5:
        written = h5["written"][:]

    mask = (segments["label_quality"].values == "strong_primary") & written
    if species is not None:
        mask &= segments["primary_label"].astype(str).values == species

    valid = np.where(mask)[0]
    if len(valid) == 0:
        raise RuntimeError(
            f"No strong_primary clips found for species='{species}'. "
            f"Try a different species or drop the filter."
        )

    rng = np.random.default_rng(seed)
    if segment_id is not None:
        matches = segments.index[segments["segment_id"] == segment_id].tolist()
        if not matches:
            raise RuntimeError(f"segment_id not found: {segment_id}")
        idx = matches[0]
    else:
        idx = int(rng.choice(valid))

    row = segments.iloc[idx]
    true_species = str(row["primary_label"])
    if true_species not in sp_to_idx:
        raise RuntimeError(
            f"Segment's primary_label {true_species} not in species_order "
            f"(likely a fallback species). Pick a different clip."
        )
    true_sp_idx = sp_to_idx[true_species]
    print(f"Clip: {row['segment_id']}")
    print(f"  true species: {true_species} (idx {true_sp_idx})")

    # ── Load the clip's mel ──────────────────────────────────────────────────
    n_mels, T = get_mel_shape(cfg)
    mels_path = emb_h5 + ".mels.npy"
    mels = np.memmap(mels_path, dtype=np.float16, mode="r",
                     shape=(len(segments), n_mels, T))
    V = np.maximum(mels[idx].astype(np.float32), 0.0)  # (n_mels, T)

    # ── Compute H for all species (one matmul) + recon errors ────────────────
    H_all = W_pinv @ V                         # (sum_k, T)
    recon_errs = _compute_species_recon_err(V, W_all, W_pinv, boundaries)

    # ── Rank species by positive-activation energy ───────────────────────────
    # pos_energy[sp] = || max(H_sp, 0) ||_F^2 — the non-negative part the
    # species' basis is legitimately "firing" for (NMF-valid contribution).
    num_species = len(boundaries) - 1
    pos_energy = np.zeros(num_species, dtype=np.float64)
    for sp_idx in range(num_species):
        c0, c1 = int(boundaries[sp_idx]), int(boundaries[sp_idx + 1])
        if c1 <= c0:
            pos_energy[sp_idx] = -np.inf  # ranked last
            continue
        H_sp = H_all[c0:c1, :]
        pos_energy[sp_idx] = float(np.sum(np.maximum(H_sp, 0.0) ** 2))

    energy_order = np.argsort(-pos_energy)     # highest energy first
    true_rank = int(np.where(energy_order == true_sp_idx)[0][0])
    wrong_candidates = [i for i in energy_order if i != true_sp_idx]
    top1_idx = wrong_candidates[0]             # highest activation energy (excl. true)
    top2_idx = wrong_candidates[1]             # 2nd-highest

    top1_name = species_order[top1_idx]
    top2_name = species_order[top2_idx]

    print(f"  true species rank by activation energy: {true_rank + 1}/{num_species}")
    print(f"  pos_energy[true]:  {pos_energy[true_sp_idx]:.3f}  recon={recon_errs[true_sp_idx]:.3f}")
    print(f"  pos_energy[top1]:  {pos_energy[top1_idx]:.3f}  recon={recon_errs[top1_idx]:.3f}  ({top1_name})")
    print(f"  pos_energy[top2]:  {pos_energy[top2_idx]:.3f}  recon={recon_errs[top2_idx]:.3f}  ({top2_name})")

    H_true = _slice_H(H_all, boundaries, true_sp_idx)
    H_top1 = _slice_H(H_all, boundaries, top1_idx)
    H_top2 = _slice_H(H_all, boundaries, top2_idx)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(10, 9),
                              gridspec_kw={"height_ratios": [2, 1.5, 1.5, 1.5]})

    # Mel spec
    ax = axes[0]
    im = ax.imshow(V, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(f"Mel spectrogram — {row['segment_id']}   (true: {true_species})")
    ax.set_ylabel("mel bin")
    fig.colorbar(im, ax=ax, fraction=0.02)

    # Shared color scale across H panels (centered at zero)
    H_stack = np.concatenate([H_true, H_top1, H_top2], axis=0)
    vmax = float(np.max(np.abs(H_stack))) + 1e-10

    for ax, H, title in zip(
        axes[1:],
        [H_true, H_top1, H_top2],
        [
            f"H_sp for TRUE species  ({true_species}, k={H_true.shape[0]}, "
            f"pos_E={pos_energy[true_sp_idx]:.1f}, rank={true_rank + 1}/{num_species}, "
            f"recon={recon_errs[true_sp_idx]:.2f}, neg_frac={(H_true<0).mean():.2f})",
            f"H_sp for #1 activation ({top1_name}, k={H_top1.shape[0]}, "
            f"pos_E={pos_energy[top1_idx]:.1f}, "
            f"recon={recon_errs[top1_idx]:.2f}, neg_frac={(H_top1<0).mean():.2f})",
            f"H_sp for #2 activation ({top2_name}, k={H_top2.shape[0]}, "
            f"pos_E={pos_energy[top2_idx]:.1f}, "
            f"recon={recon_errs[top2_idx]:.2f}, neg_frac={(H_top2<0).mean():.2f})",
        ],
    ):
        im = ax.imshow(H, aspect="auto", origin="lower", cmap="seismic",
                       vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_ylabel("H component")
        fig.colorbar(im, ax=ax, fraction=0.02)
    axes[-1].set_xlabel("time frame")

    fig.suptitle(
        "Pseudoinverse activations H = W_pinv @ V per species\n"
        "Positive = basis atom fires; Negative = least-squares had to subtract "
        "to fit (basis doesn't belong).",
        y=1.01,
    )
    fig.tight_layout()

    if html:
        if not out_path:
            raise ValueError("--html requires --out path (e.g. foo.html)")
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        png_bytes = _figure_to_png_bytes(fig)

        # Load raw audio to embed next to the figure
        sr = cfg["data"]["sample_rate"]
        wav = load_audio_segment(
            str(row["source_file"]),
            float(row["start_sec"]),
            float(row["end_sec"]),
            sr=sr,
        )
        wav_bytes = _wav_to_wav_bytes(wav.astype(np.float32), sr)
        duration = float(row["end_sec"]) - float(row["start_sec"])

        title = f"{row['segment_id']}  (true: {true_species})"
        _write_html_page(png_bytes, wav_bytes, title, duration, V.shape[1], out_p)
        print(f"Saved → {out_p}  (open in a browser)")

    elif out_path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_p}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", type=str, default=None,
                        help="primary_label to filter on (e.g. 'baymac'). Default: any")
    parser.add_argument("--segment-id", type=str, default=None,
                        help="specific segment_id to visualize (overrides --species sampling)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None,
                        help="output .png or .html path; if omitted, shows interactive")
    parser.add_argument("--html", action="store_true",
                        help="emit a self-contained HTML page with audio player "
                              "and a time cursor synced to playback")
    args = parser.parse_args()

    cfg = get_config()
    visualize_one_clip(cfg, args.species, args.segment_id, args.out,
                       args.seed, html=args.html)


if __name__ == "__main__":
    main()
