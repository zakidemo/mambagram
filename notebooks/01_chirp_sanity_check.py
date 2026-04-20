"""
Sanity check for MambaGramLayer.

Generates a synthetic linear chirp, runs it through MambaGramLayer at
initialization (no training), and plots the resulting magnitude
representation |h| alongside a standard STFT magnitude for comparison.

Success criterion (visual): |h| shows a clear diagonal ridge tracking the
chirp's instantaneous frequency, similar to the STFT. If the MambaGram
ridge is visible and roughly matches the STFT ridge, the math is correct.

Run from project root:
    python notebooks/01_chirp_sanity_check.py
"""
import math
import os
import sys
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mambagram import MambaGramLayer


def make_linear_chirp(
    duration_s: float = 2.0,
    fs: int = 16000,
    f0: float = 200.0,
    f1: float = 4000.0,
) -> torch.Tensor:
    """Linear chirp from f0 Hz to f1 Hz over duration_s seconds."""
    t = torch.arange(0, int(duration_s * fs)) / fs
    # instantaneous phase for linear chirp
    phase = 2 * math.pi * (f0 * t + 0.5 * (f1 - f0) / duration_s * t ** 2)
    x = torch.sin(phase).float()
    return x


def stft_magnitude(x: torch.Tensor, n_fft: int = 512, hop: int = 128) -> torch.Tensor:
    """Standard STFT magnitude for comparison."""
    window = torch.hann_window(n_fft)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    return X.abs()  # (freq, time)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- 1. Generate chirp ---
    fs = 16000
    duration = 2.0
    x = make_linear_chirp(duration_s=duration, fs=fs, f0=200.0, f1=4000.0)
    print(f"Chirp length: {x.shape[0]} samples ({duration}s @ {fs}Hz)")

    # Add batch dim
    x_batch = x.unsqueeze(0).to(device)  # (1, L)

    # --- 2. Compute MambaGram ---
    layer = MambaGramLayer(
        n_channels=64,
        sample_rate=fs,
        f_min=80.0,
        f_max=fs / 2,
        window_ms=25.0,
        init="mel",
    ).to(device)

    print(f"MambaGram channels: {layer.n_channels}")
    print(f"Initial omega range: [{layer.omega.min().item():.4f}, {layer.omega.max().item():.4f}] rad/sample")
    print(f"Initial freq range:  [{(layer.omega.min() * fs / (2 * math.pi)).item():.1f}, "
          f"{(layer.omega.max() * fs / (2 * math.pi)).item():.1f}] Hz")

    t0 = time.time()
    with torch.no_grad():
        mag = layer.magnitude(x_batch)  # (1, L, D)
    print(f"MambaGram forward pass: {time.time() - t0:.2f}s")

    mag_np = mag[0].cpu().numpy().T  # (D, L)

    # --- 3. Compute STFT for comparison ---
    stft_mag = stft_magnitude(x, n_fft=512, hop=128).numpy()  # (freq, time)

    # --- 4. Plot side-by-side ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    # STFT
    axes[0].imshow(
        20 * np.log10(stft_mag + 1e-8),
        aspect="auto",
        origin="lower",
        extent=[0, duration, 0, fs / 2],
        cmap="magma",
    )
    axes[0].set_title("Standard STFT magnitude (dB)")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_xlabel("Time (s)")

    # MambaGram
    freqs_mg = (layer.omega.detach().cpu().numpy() * fs) / (2 * math.pi)
    axes[1].imshow(
        20 * np.log10(mag_np + 1e-8),
        aspect="auto",
        origin="lower",
        extent=[0, duration, freqs_mg.min(), freqs_mg.max()],
        cmap="magma",
    )
    axes[1].set_title("MambaGram |h| (dB) — at initialization, no training")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_xlabel("Time (s)")

    plt.tight_layout()
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "figures", "01_chirp_sanity_check.png"
    )
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved figure to: {out_path}")

    print("\n--- Sanity check complete ---")
    print("Look at the figure: both plots should show a diagonal ridge rising")
    print("from ~200 Hz to ~4000 Hz. If yes, MambaGramLayer is working correctly.")


if __name__ == "__main__":
    main()
