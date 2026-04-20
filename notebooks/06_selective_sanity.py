"""
Step 9: Sanity check for SelectiveMambaGramLayer.

At initialization, Delta_t = 1.0 everywhere by design (selective layer
behaves like baseline). To demonstrate that the architecture CAN produce
input-dependent behavior, we manually set the MLP weights to make
Delta_t large on quiet (low-amplitude) regions and small on loud regions.
This is the kind of behavior we expect training to learn automatically.

We then visualize:
  - The input signal (a chirp + click composite)
  - Delta_t trajectory
  - MambaGram |h| with constant Delta=1 (baseline)
  - MambaGram |h| with input-dependent Delta (selective)

The selective version should show sharper time localization at the click
because Delta contracts there.
"""
import math
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mambagram import MambaGramLayer, SelectiveMambaGramLayer


def make_chirp_with_click(duration_s=2.0, fs=16000):
    """A 200 -> 4000 Hz chirp with a sharp click superimposed at t=1.0s."""
    t = torch.arange(0, int(duration_s * fs)) / fs
    chirp_phase = 2 * math.pi * (200.0 * t + 0.5 * (4000 - 200) / duration_s * t**2)
    chirp = 0.5 * torch.sin(chirp_phase).float()
    click = 1.5 * torch.exp(-0.5 * ((t - 1.0) / 0.001) ** 2)
    return (chirp + click).float()


def to_db_norm(mag, eps=1e-8):
    mag = mag / (mag.max() + eps)
    return 20 * np.log10(mag.clip(min=eps))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = 16000
    duration = 2.0
    x = make_chirp_with_click(duration, fs)
    x_batch = x.unsqueeze(0).to(device)

    # --- Baseline (constant A) ---
    baseline = MambaGramLayer(n_channels=64, sample_rate=fs,
                              f_min=80.0, f_max=fs/2,
                              window_ms=25.0).to(device)
    with torch.no_grad():
        mag_baseline = baseline.magnitude(x_batch)[0].cpu().numpy().T  # (D, L)

    # --- Selective layer at initialization (Delta = 1.0 everywhere) ---
    selective = SelectiveMambaGramLayer(n_channels=64, sample_rate=fs,
                                         f_min=80.0, f_max=fs/2,
                                         window_ms=25.0,
                                         delta_min=0.1, delta_max=5.0).to(device)

    # Hand-craft selectivity: make Delta_t small where |x| is large.
    # Specifically, set the MLP to compute Delta_t = 5 * exp(-3 * |x|).
    # (This is what we hope training would discover for transients.)
    # We do this by directly manipulating the MLP outputs in a hook,
    # rather than retraining — this is just a demonstration.
    with torch.no_grad():
        h_init, delta_init = selective(x_batch, return_delta=True)
    delta_init_np = delta_init[0].cpu().numpy()

    # Manual demonstration of input-dependent Delta
    # We bypass the MLP and inject our own Delta trajectory.
    class HandCraftedSelective(SelectiveMambaGramLayer):
        def _compute_delta(self, x):
            # Big Delta on quiet regions (sharper freq), small Delta on loud (sharper time)
            amp = x.abs()
            # Normalize amp to [0, 1] over the whole batch
            amp_norm = amp / (amp.max() + 1e-8)
            # Delta in [delta_min, delta_max], inversely proportional to amp
            delta = self.delta_min + (self.delta_max - self.delta_min) * (1.0 - amp_norm)
            return delta

    handcrafted = HandCraftedSelective(n_channels=64, sample_rate=fs,
                                        f_min=80.0, f_max=fs/2,
                                        window_ms=25.0,
                                        delta_min=0.1, delta_max=5.0).to(device)
    # Copy parameters from `selective` so only Delta differs
    handcrafted.omega = selective.omega
    handcrafted.raw_alpha = selective.raw_alpha
    handcrafted.b_real = selective.b_real
    handcrafted.b_imag = selective.b_imag

    with torch.no_grad():
        h_hand, delta_hand = handcrafted(x_batch, return_delta=True)
    mag_hand = h_hand[0].abs().cpu().numpy().T  # (D, L)
    delta_hand_np = delta_hand[0].cpu().numpy()

    # --- Plot ---
    fig, axes = plt.subplots(4, 1, figsize=(11, 12))
    t_axis = np.arange(len(x)) / fs

    # Panel 1: input signal
    axes[0].plot(t_axis, x.numpy(), lw=0.5, color="steelblue")
    axes[0].set_title("Input: chirp (200→4000 Hz) + click @ 1.0s")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, duration)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Delta_t trajectories
    axes[1].plot(t_axis, np.full_like(t_axis, 1.0), "k--",
                 label="Baseline: Δ=1 (constant)", alpha=0.5)
    axes[1].plot(t_axis, delta_init_np, "g-",
                 label="Selective init: Δ≈1 everywhere", alpha=0.7)
    axes[1].plot(t_axis, delta_hand_np, "r-",
                 label="Hand-crafted selective: Δ small at click", lw=1.5)
    axes[1].set_title("Δ_t trajectory: how the effective window length adapts")
    axes[1].set_ylabel("Δ_t (timescale)")
    axes[1].set_xlim(0, duration)
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

    # Panel 4: hand-crafted selective MambaGram
    im3 = axes[3].imshow(to_db_norm(mag_hand),
                         aspect="auto", origin="lower",
                         extent=[0, duration, 0, 64],
                         cmap="magma", vmin=-50, vmax=0)
    axes[3].set_title("Hand-crafted selective MambaGram |h|: Δ contracts at the click")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Channel index d")
    plt.colorbar(im3, ax=axes[3], label="dB")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures",
                       "06_selective_sanity.png")
    plt.savefig(out, dpi=130)
    print(f"Saved: {out}")

    # Quantify: how narrow is the click in each version?
    # Take a horizontal slice at channel near 4000 Hz (~ch 50 in mel)
    ch = 50
    base_slice = mag_baseline[ch]
    hand_slice = mag_hand[ch]
    # Find FWHM of the click peak around t=1.0s
    click_idx = int(1.0 * fs)
    win = 2000  # +/- 0.125s
    base_local = base_slice[click_idx - win:click_idx + win]
    hand_local = hand_slice[click_idx - win:click_idx + win]
    print(f"\nClick peak width (in samples, FWHM, channel {ch}):")
    print(f"  Baseline:    peak={base_local.max():.4f}, "
          f"width@half-max={(base_local > base_local.max()/2).sum()}")
    print(f"  Hand-crafted: peak={hand_local.max():.4f}, "
          f"width@half-max={(hand_local > hand_local.max()/2).sum()}")
    print("Smaller width = sharper time localization (selectivity working).")


if __name__ == "__main__":
    main()
