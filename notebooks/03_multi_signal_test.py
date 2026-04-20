"""
Step 6: Multi-signal sanity test for MambaGramLayer.

Evaluates MambaGram on three signal archetypes, each of which is
traditionally handled best by a different TF representation:

  1. Linear chirp         (STFT's home domain)
  2. Gaussian click       (wavelet scalogram's home domain)
  3. Harmonic tone stack  (CQT's home domain)

Each signal is processed through the same, untrained MambaGram layer
(D=64 channels, Mel-spaced) with no parameter changes between signals.
This is the precursor to the paper's main figure showing that
MambaGram adapts to signal content.

Outputs:
    figures/03_multi_signal_test.png  — 3x2 grid: each row a signal,
        left column = waveform, right column = MambaGram |h|.
"""
import math
import os
import sys
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mambagram import MambaGramLayer


# --- Signal generators ---

def make_linear_chirp(duration_s=2.0, fs=16000, f0=200.0, f1=4000.0):
    t = torch.arange(0, int(duration_s * fs)) / fs
    phase = 2 * math.pi * (f0 * t + 0.5 * (f1 - f0) / duration_s * t**2)
    return torch.sin(phase).float()


def make_click(duration_s=2.0, fs=16000, click_time_s=1.0, width_s=0.002):
    """Gaussian click centered at click_time_s."""
    t = torch.arange(0, int(duration_s * fs)) / fs
    click = torch.exp(-0.5 * ((t - click_time_s) / width_s) ** 2)
    # Add a little noise floor so the plot isn't completely dark
    noise = 0.01 * torch.randn_like(click)
    return (click + noise).float()


def make_harmonic_stack(duration_s=2.0, fs=16000, f0=220.0, n_harmonics=6):
    """Sum of harmonics at f0, 2*f0, 3*f0, ... simulating a musical note."""
    t = torch.arange(0, int(duration_s * fs)) / fs
    y = torch.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        if k * f0 >= fs / 2:
            break
        # Decreasing amplitude for higher harmonics (more realistic)
        y += (1.0 / k) * torch.sin(2 * math.pi * k * f0 * t)
    # Normalize to unit amplitude
    y = y / y.abs().max()
    return y.float()


# --- Plotting helper ---

def to_db_normalized(mag, eps=1e-8):
    mag_n = mag / (mag.max() + eps)
    return 20 * np.log10(mag_n + eps)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = 16000
    duration = 2.0

    # --- Generate signals ---
    signals = {
        "Linear chirp (200 → 4000 Hz)": make_linear_chirp(duration, fs, 200.0, 4000.0),
        "Gaussian click @ 1.0 s": make_click(duration, fs, click_time_s=1.0, width_s=0.002),
        "Harmonic stack (f₀=220 Hz, 6 harmonics)": make_harmonic_stack(duration, fs, 220.0, 6),
    }

    # --- Shared MambaGram layer (same parameters for all three signals) ---
    layer = MambaGramLayer(
        n_channels=64, sample_rate=fs,
        f_min=80.0, f_max=fs / 2,
        window_ms=25.0, init="mel",
    ).to(device)
    freqs_mg = (layer.omega.detach().cpu().numpy() * fs) / (2 * math.pi)

    print(f"MambaGram: D={layer.n_channels}, window={layer.window_ms}ms, device={device}")
    print(f"Channel freqs: {freqs_mg[0]:.0f}–{freqs_mg[-1]:.0f} Hz (Mel-spaced)\n")

    # --- Plot: 3 rows x 2 cols (waveform | MambaGram) ---
    fig, axes = plt.subplots(len(signals), 2, figsize=(12, 9))

    for i, (name, x) in enumerate(signals.items()):
        t_axis = np.arange(len(x)) / fs
        x_batch = x.unsqueeze(0).to(device)

        t0 = time.time()
        with torch.no_grad():
            mag = layer.magnitude(x_batch)[0]  # (L, D)
        dt = time.time() - t0
        mag_np = mag.cpu().numpy().T  # (D, L)

        print(f"[{i+1}/3] {name}")
        print(f"    waveform: length={len(x)}, peak={x.abs().max():.3f}")
        print(f"    MambaGram forward: {dt:.2f}s")
        print(f"    |h| range: [{mag_np.min():.4f}, {mag_np.max():.4f}]\n")

        # Waveform (left)
        axes[i, 0].plot(t_axis, x.numpy(), lw=0.6, color="steelblue")
        axes[i, 0].set_title(f"{name} — waveform")
        axes[i, 0].set_xlabel("Time (s)")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_xlim(0, duration)

        # MambaGram (right)
        im = axes[i, 1].imshow(
            to_db_normalized(mag_np),
            aspect="auto", origin="lower",
            extent=[0, duration, 0, layer.n_channels - 1],
            cmap="magma", vmin=-60, vmax=0,
        )
        axes[i, 1].set_title(f"{name} — MambaGram |h|")
        axes[i, 1].set_xlabel("Time (s)")
        axes[i, 1].set_ylabel("Channel index d")
        tick_idx = [0, 16, 32, 48, 63]
        axes[i, 1].set_yticks(tick_idx)
        axes[i, 1].set_yticklabels([f"{freqs_mg[j]:.0f}" for j in tick_idx])
        plt.colorbar(im, ax=axes[i, 1], label="dB (norm)")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "03_multi_signal_test.png")
    plt.savefig(out, dpi=120)
    print(f"Saved: {out}\n")

    print("--- Interpretation guide ---")
    print("  Chirp  : expect a clean diagonal ridge (same as Step 5).")
    print("  Click  : expect a vertical stripe at t=1.0s across many channels —")
    print("           this proves MambaGram correctly resolves transients.")
    print("  Harmonic: expect horizontal bands at 220, 440, 660, ... Hz —")
    print("           this proves MambaGram resolves stable tonal content.")
    print("\nAll three behaviors come from the SAME layer parameters,")
    print("illustrating that MambaGram adapts its output to signal content.")


if __name__ == "__main__":
    main()
