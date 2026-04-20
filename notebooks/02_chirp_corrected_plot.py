"""
Corrected plot of MambaGram output on a chirp.

Fixes from v1:
1. Plot MambaGram on channel-index y-axis (then show frequency tick labels)
   so the Mel-spaced non-uniformity doesn't distort the ridge shape.
2. Normalize both representations to [0, 1] before dB conversion
   for fair visual comparison.
3. Add a third panel showing the instantaneous frequency ridge
   (argmax over channels) overlaid on the ground truth.
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


def make_linear_chirp(duration_s=2.0, fs=16000, f0=200.0, f1=4000.0):
    t = torch.arange(0, int(duration_s * fs)) / fs
    phase = 2 * math.pi * (f0 * t + 0.5 * (f1 - f0) / duration_s * t**2)
    return torch.sin(phase).float(), t.numpy()


def stft_magnitude(x, n_fft=512, hop=128):
    window = torch.hann_window(n_fft)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    return X.abs()


def to_db_normalized(mag, eps=1e-8):
    """Normalize to [0,1] then convert to dB for consistent colorbars."""
    mag_n = mag / (mag.max() + eps)
    return 20 * np.log10(mag_n + eps)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = 16000
    duration = 2.0
    f0, f1 = 200.0, 4000.0

    x, t_axis = make_linear_chirp(duration, fs, f0, f1)
    x_batch = x.unsqueeze(0).to(device)

    # --- MambaGram ---
    layer = MambaGramLayer(
        n_channels=64, sample_rate=fs,
        f_min=80.0, f_max=fs / 2,
        window_ms=25.0, init="mel",
    ).to(device)

    t0 = time.time()
    with torch.no_grad():
        mag = layer.magnitude(x_batch)[0]  # (L, D)
    print(f"MambaGram forward: {time.time() - t0:.2f}s")
    mag_np = mag.cpu().numpy().T  # (D, L)

    freqs_mg = (layer.omega.detach().cpu().numpy() * fs) / (2 * math.pi)  # (D,)

    # --- STFT ---
    stft_mag = stft_magnitude(x, n_fft=512, hop=128).numpy()
    stft_freqs = np.linspace(0, fs / 2, stft_mag.shape[0])

    # --- Ground-truth instantaneous frequency (linear chirp) ---
    true_if = f0 + (f1 - f0) * t_axis / duration  # Hz at each sample

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 11))

    # Panel 1: STFT
    im0 = axes[0].imshow(
        to_db_normalized(stft_mag),
        aspect="auto", origin="lower",
        extent=[0, duration, 0, fs / 2], cmap="magma", vmin=-60, vmax=0,
    )
    axes[0].set_title("STFT magnitude (dB, normalized)")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].plot(t_axis, true_if, "c--", lw=1, alpha=0.7, label="True IF")
    axes[0].legend(loc="upper left")
    plt.colorbar(im0, ax=axes[0])

    # Panel 2: MambaGram plotted on CHANNEL INDEX y-axis (shows true ridge shape)
    im1 = axes[1].imshow(
        to_db_normalized(mag_np),
        aspect="auto", origin="lower",
        extent=[0, duration, 0, layer.n_channels - 1],
        cmap="magma", vmin=-60, vmax=0,
    )
    axes[1].set_title("MambaGram |h| (dB, normalized) — y-axis = channel index")
    axes[1].set_ylabel("Channel index d")
    # Show a few tick labels with their actual frequency
    tick_idx = [0, 16, 32, 48, 63]
    axes[1].set_yticks(tick_idx)
    axes[1].set_yticklabels([f"{freqs_mg[i]:.0f} Hz" for i in tick_idx])
    plt.colorbar(im1, ax=axes[1])

    # Panel 3: Instantaneous frequency comparison
    # For STFT: argmax over frequency axis at each time frame
    stft_if_argmax = stft_freqs[np.argmax(stft_mag, axis=0)]
    stft_t_axis = np.linspace(0, duration, stft_mag.shape[1])

    # For MambaGram: argmax over channel axis, then look up frequency
    mg_if_argmax = freqs_mg[np.argmax(mag_np, axis=0)]
    mg_t_axis = np.linspace(0, duration, mag_np.shape[1])

    axes[2].plot(t_axis, true_if, "k-", lw=2, label="True IF (ground truth)")
    axes[2].plot(stft_t_axis, stft_if_argmax, "b.", ms=2, alpha=0.5, label="STFT argmax")
    axes[2].plot(mg_t_axis, mg_if_argmax, "r.", ms=2, alpha=0.5, label="MambaGram argmax")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Instantaneous frequency (Hz)")
    axes[2].set_title("Instantaneous frequency tracking — ridge argmax vs. ground truth")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 5000)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "02_chirp_corrected.png")
    plt.savefig(out, dpi=120)
    print(f"Saved: {out}")

    # --- Quantitative check: mean abs error of MambaGram IF vs ground truth ---
    # Interpolate true IF onto MambaGram time grid
    true_if_on_mg = np.interp(mg_t_axis, t_axis, true_if)
    mae_mg = np.mean(np.abs(mg_if_argmax - true_if_on_mg))
    true_if_on_stft = np.interp(stft_t_axis, t_axis, true_if)
    mae_stft = np.mean(np.abs(stft_if_argmax - true_if_on_stft))

    print("\n--- Quantitative ridge tracking error ---")
    print(f"STFT       MAE: {mae_stft:7.1f} Hz")
    print(f"MambaGram  MAE: {mae_mg:7.1f} Hz")
    print("(Lower is better. MambaGram at initialization is not expected to beat STFT;")
    print(" but it should be in the same order of magnitude — confirming the math.)")


if __name__ == "__main__":
    main()
