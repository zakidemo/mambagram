"""
Step 9 diagnostic: is the GatedMambaGramLayer correct?

We bypass all amplitude-based gating and inject a hand-crafted
TIME-BASED gate:

    gamma(t) = 1.0            everywhere,
    EXCEPT near t = 1.0 s where gamma drops smoothly to 0.05
    (over a window of +/- 10 ms, Gaussian shape).

This gate is:
  - a simple function of t only (no dependency on x)
  - guaranteed smooth in time (Gaussian is infinitely differentiable)
  - gamma = 1.0 for 99.9% of the signal

If this gate PRESERVES the chirp ridge AND sharpens the click, the
GatedMambaGramLayer math is correct and the earlier failure was due
to a poor amplitude-based gate formula.

If this gate ALSO fails, the gated-layer math has a deeper bug
and we need to audit it.
"""
import math
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from mambagram import MambaGramLayer, GatedMambaGramLayer


def make_chirp_with_click(duration_s=2.0, fs=16000):
    t = torch.arange(0, int(duration_s * fs)) / fs
    chirp_phase = 2 * math.pi * (200.0 * t + 0.5 * (4000 - 200) / duration_s * t ** 2)
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

    # Baseline
    baseline = MambaGramLayer(n_channels=64, sample_rate=fs,
                              f_min=80.0, f_max=fs / 2,
                              window_ms=25.0).to(device)
    with torch.no_grad():
        mag_baseline = baseline.magnitude(x_batch)[0].cpu().numpy().T

    # Gated layer with a PURELY TIME-BASED gate
    class TimeBasedGate(GatedMambaGramLayer):
        def _compute_gate(self, x):
            # Gate = 1.0 - 0.95 * Gaussian pulse at t = 1.0 s, width 3 ms
            batch, length = x.shape
            t = torch.arange(length, device=x.device).float() / self.sample_rate
            click_t = 1.0
            sigma = 3e-3  # 3 ms standard deviation
            pulse = torch.exp(-0.5 * ((t - click_t) / sigma) ** 2)  # (L,)
            gate = 1.0 - 0.95 * pulse                               # (L,)
            return gate.unsqueeze(0).expand(batch, -1)              # (B, L)

    gated = TimeBasedGate(n_channels=64, sample_rate=fs,
                          f_min=80.0, f_max=fs / 2,
                          window_ms=25.0,
                          gate_smooth_ms=0.0).to(device)

    # Copy baseline parameters so ONLY the gate differs
    with torch.no_grad():
        gated.omega.copy_(baseline.omega)
        gated.raw_alpha.copy_(baseline.raw_alpha)
        gated.b_real.copy_(baseline.b_real)
        gated.b_imag.copy_(baseline.b_imag)

    with torch.no_grad():
        h_gated, gate_trace = gated(x_batch, return_gate=True)
    mag_gated = h_gated[0].abs().cpu().numpy().T
    gate_np = gate_trace[0].cpu().numpy()

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(11, 12))
    t_axis = np.arange(len(x)) / fs

    axes[0].plot(t_axis, x.numpy(), lw=0.5, color="steelblue")
    axes[0].set_title("Input: chirp + click at t=1.0 s")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, duration)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_axis, np.ones_like(t_axis), "k--",
                 label="Baseline (gamma=1)", lw=2, alpha=0.5)
    axes[1].plot(t_axis, gate_np, "r-",
                 label="Time-based gate: Gaussian dip at t=1 s", lw=2)
    axes[1].set_title("Hand-crafted TIME-BASED gate (not derived from x)")
    axes[1].set_ylabel("gamma_t")
    axes[1].set_ylim(-0.05, 1.1)
    axes[1].set_xlim(0, duration)
    axes[1].legend(loc="center right")
    axes[1].grid(True, alpha=0.3)

    im2 = axes[2].imshow(to_db_norm(mag_baseline),
                         aspect="auto", origin="lower",
                         extent=[0, duration, 0, 64],
                         cmap="magma", vmin=-50, vmax=0)
    axes[2].set_title("Baseline MambaGram |h|")
    axes[2].set_ylabel("Channel index d")
    plt.colorbar(im2, ax=axes[2], label="dB")

    im3 = axes[3].imshow(to_db_norm(mag_gated),
                         aspect="auto", origin="lower",
                         extent=[0, duration, 0, 64],
                         cmap="magma", vmin=-50, vmax=0)
    axes[3].set_title("Time-gated MambaGram |h| - does chirp survive? does click sharpen?")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Channel index d")
    plt.colorbar(im3, ax=axes[3], label="dB")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures",
                       "08_gate_diagnostic.png")
    plt.savefig(out, dpi=130)
    print(f"Saved: {out}")

    # Metrics
    ch = 50
    click_idx = int(1.0 * fs)
    win = 2000
    b_local = mag_baseline[ch, click_idx - win:click_idx + win]
    g_local = mag_gated[ch, click_idx - win:click_idx + win]

    print(f"\nClick width at channel {ch}:")
    print(f"  Baseline: peak={b_local.max():.4f}, "
          f"FWHM={(b_local > b_local.max()/2).sum()} samples")
    print(f"  Gated:    peak={g_local.max():.4f}, "
          f"FWHM={(g_local > g_local.max()/2).sum()} samples")

    ridge_t = int(0.5 * fs)
    b_col = mag_baseline[:, ridge_t]
    g_col = mag_gated[:, ridge_t]
    print(f"\nChirp ridge at t=0.5s:")
    print(f"  Baseline: peak channel={b_col.argmax()}, "
          f"peak/mean ratio={b_col.max()/b_col.mean():.2f}")
    print(f"  Gated:    peak channel={g_col.argmax()}, "
          f"peak/mean ratio={g_col.max()/g_col.mean():.2f}")

    # Since gate = 1 at t = 0.5s, the gated panel SHOULD look essentially
    # identical to baseline there. If it doesn't, the layer math is broken.
    chirp_identical = (
        int(b_col.argmax()) == int(g_col.argmax())
        and abs(float(b_col.max()) - float(g_col.max())) / float(b_col.max()) < 0.05
    )
    click_sharper = (g_local > g_local.max()/2).sum() < (b_local > b_local.max()/2).sum()

    print("\n--- Diagnostic verdict ---")
    print(f"  Chirp at t=0.5s identical to baseline (gate=1 there): {chirp_identical}")
    print(f"  Click is sharper with gating:                          {click_sharper}")
    if chirp_identical and click_sharper:
        print("  RESULT: Layer math is CORRECT. Previous failure was gate-formula bug.")
    elif not chirp_identical:
        print("  RESULT: Layer math is BROKEN. Even a gate=1 region is being corrupted.")
    else:
        print("  RESULT: Layer math is correct but click sharpening needs stronger gate.")


if __name__ == "__main__":
    main()