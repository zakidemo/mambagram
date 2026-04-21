"""
Step 9 (revised): Sanity check for SelectiveMambaGramLayer.

Demonstrates that smooth, input-dependent Delta_t produces sharper time
localization on transients while preserving frequency resolution
elsewhere. Lesson learned from v1: Delta_t must vary SMOOTHLY in time
or the time-varying recurrence destroys the Gabor interpretation.

We:
  1. Generate a chirp + click composite.
  2. Compute a SMOOTHED amplitude envelope via low-pass filtering |x|.
  3. Map the envelope to Delta_t: large where amplitude is small (sharp
     frequency), small where amplitude is large (sharp time at the click).
  4. Compare baseline (Delta=1) vs smoothed-selective MambaGram outputs.

Success criterion: in the smoothed-selective version, the click is
visibly narrower in time AND the chirp ridge remains intact.
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from mambagram import MambaGramLayer, SelectiveMambaGramLayer


def make_chirp_with_click(duration_s=2.0, fs=16000):
    t = torch.arange(0, int(duration_s * fs)) / fs
    chirp_phase = 2 * math.pi * (200.0 * t + 0.5 * (4000 - 200) / duration_s * t**2)
    chirp = 0.5 * torch.sin(chirp_phase).float()
    click = 1.5 * torch.exp(-0.5 * ((t - 1.0) / 0.001) ** 2)
    return (chirp + click).float()


def smoothed_envelope(x: torch.Tensor, fs: int, smooth_ms: float = 5.0) -> torch.Tensor:
    """
    Compute a smoothed amplitude envelope of x via a moving-average filter.

    Parameters
    ----------
    x       : (batch, length) tensor.
    fs      : sample rate.
    smooth_ms : averaging window in milliseconds.
    """
    win = max(1, int(smooth_ms * 1e-3 * fs))
    # Make a centered moving-average kernel
    kernel = torch.ones(1, 1, win, device=x.device) / win
    # Reflect-pad to keep length, then conv
    pad = win // 2
    x_abs = x.abs().unsqueeze(1)                                  # (B, 1, L)
    x_padded = F.pad(x_abs, (pad, pad), mode="reflect")
    env = F.conv1d(x_padded, kernel)[:, 0, :x.size(-1)]            # (B, L)
    return env


def to_db_norm(mag, eps=1e-8):
    mag = mag / (mag.max() + eps)
    return 20 * np.log10(mag.clip(min=eps))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = 16000
    duration = 2.0

    x = make_chirp_with_click(duration, fs)
    x_batch = x.unsqueeze(0).to(device)

    # --- Baseline (constant Delta=1) ---
    baseline = MambaGramLayer(n_channels=64, sample_rate=fs,
                              f_min=80.0, f_max=fs / 2,
                              window_ms=25.0).to(device)
    with torch.no_grad():
        mag_baseline = baseline.magnitude(x_batch)[0].cpu().numpy().T

    # --- Smoothed selective: Delta(x) using smoothed envelope ---
    delta_min, delta_max = 0.2, 4.0

    class SmoothedSelective(SelectiveMambaGramLayer):
        def _compute_delta(self, x):
            # Smoothed amplitude envelope (5 ms moving avg)
            env = smoothed_envelope(x, fs=fs, smooth_ms=5.0)        # (B, L)
            # Normalize per-batch to [0, 1]
            env_norm = env / (env.max(dim=-1, keepdim=True).values + 1e-8)
            # Delta: small at high amplitude, large at low amplitude
            delta = self.delta_min + (self.delta_max - self.delta_min) * (1.0 - env_norm)
            return delta

    smoothed = SmoothedSelective(n_channels=64, sample_rate=fs,
                                  f_min=80.0, f_max=fs / 2,
                                  window_ms=25.0,
                                  delta_min=delta_min, delta_max=delta_max).to(device)

    with torch.no_grad():
        h_sel, delta_sel = smoothed(x_batch, return_delta=True)
    mag_sel = h_sel[0].abs().cpu().numpy().T
    delta_sel_np = delta_sel[0].cpu().numpy()

    # Also: get the unmodified selective layer's Delta for comparison
    selective = SelectiveMambaGramLayer(n_channels=64, sample_rate=fs,
                                         f_min=80.0, f_max=fs / 2,
                                         window_ms=25.0,
                                         delta_min=delta_min,
                                         delta_max=delta_max).to(device)
    with torch.no_grad():
        _, delta_init = selective(x_batch, return_delta=True)
    delta_init_np = delta_init[0].cpu().numpy()

    # --- Plot ---
    fig, axes = plt.subplots(4, 1, figsize=(11, 12))
    t_axis = np.arange(len(x)) / fs

    # Panel 1: input
    axes[0].plot(t_axis, x.numpy(), lw=0.5, color="steelblue")
    axes[0].set_title("Input: chirp (200→4000 Hz) + sharp click @ 1.0s")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, duration)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Delta trajectories (now smoothed)
    axes[1].plot(t_axis, np.full_like(t_axis, 1.0), "k--",
                 label="Baseline (constant Δ=1)", alpha=0.5, lw=2)
    axes[1].plot(t_axis, delta_init_np, "g-",
                 label="Selective at init (≈1.0 by design)", alpha=0.8, lw=1.5)
    axes[1].plot(t_axis, delta_sel_np, "r-",
                 label="Smoothed-envelope Δ(x): contracts at click", lw=2)
    axes[1].set_title("Δ_t trajectory (smoothed via 5 ms moving average)")
    axes[1].set_ylabel("Δ_t")
    axes[1].set_xlim(0, duration)
    axes[1].set_ylim(0, delta_max + 0.5)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: baseline MambaGram
    im2 = axes[2].imshow(to_db_norm(mag_baseline),
                         aspect="auto", origin="lower",
                         extent=[0, duration, 0, 64],
                         cmap="magma", vmin=-50, vmax=0)
    axes[2].set_title("Baseline MambaGram |h| (constant Δ=1)")
    axes[2].set_ylabel("Channel index d")
    plt.colorbar(im2, ax=axes[2], label="dB")

    # Panel 4: smoothed selective MambaGram
    im3 = axes[3].imshow(to_db_norm(mag_sel),
                         aspect="auto", origin="lower",
                         extent=[0, duration, 0, 64],
                         cmap="magma", vmin=-50, vmax=0)
    axes[3].set_title("Smoothed-selective MambaGram |h| — sharper click")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Channel index d")
    plt.colorbar(im3, ax=axes[3], label="dB")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures",
                       "06_selective_sanity_v2.png")
    plt.savefig(out, dpi=130)
    print(f"Saved: {out}")

    # --- Quantitative: width of click in each version ---
    # Pick a high-frequency channel that responds to the click
    ch = 50
    base_slice = mag_baseline[ch]
    sel_slice = mag_sel[ch]

    click_idx = int(1.0 * fs)
    win = 2000
    base_local = base_slice[click_idx - win:click_idx + win]
    sel_local = sel_slice[click_idx - win:click_idx + win]

    print(f"\nClick width measurement (channel {ch}, +/- 0.125s around t=1.0s):")
    print(f"  Baseline:  peak={base_local.max():.4f}, "
          f"FWHM={(base_local > base_local.max()/2).sum()} samples")
    print(f"  Selective: peak={sel_local.max():.4f}, "
          f"FWHM={(sel_local > sel_local.max()/2).sum()} samples")

    # Also check chirp ridge integrity
    # at t=0.5s, where chirp should be at ~1100 Hz (channel ~25 in mel)
    ridge_t = int(0.5 * fs)
    base_col = mag_baseline[:, ridge_t]
    sel_col = mag_sel[:, ridge_t]
    print(f"\nChirp ridge sharpness at t=0.5s (peakedness in channel space):")
    print(f"  Baseline:  peak channel={base_col.argmax()}, peak/mean ratio={base_col.max()/base_col.mean():.2f}")
    print(f"  Selective: peak channel={sel_col.argmax()}, peak/mean ratio={sel_col.max()/sel_col.mean():.2f}")
    print("Higher ratio = sharper ridge.")


if __name__ == "__main__":
    main()
