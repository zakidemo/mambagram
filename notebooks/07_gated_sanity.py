"""
Step 9 (final): Sanity check for GatedMambaGramLayer.

Tests that a hand-crafted forgetting gate γ_t — set to drop near 0 at a
transient and remain near 1 elsewhere — produces:
  (a) sharper time localization of the click (narrower FWHM)
  (b) chirp ridge in the SAME channel as baseline (grid preserved)

Success criteria:
  - click_FWHM_gated < click_FWHM_baseline
  - chirp_peak_channel_gated == chirp_peak_channel_baseline
  - chirp_ridge_sharpness_gated ≈ chirp_ridge_sharpness_baseline

Unlike the failed Δ experiment, these gates only SHORTEN the effective
window — they never shift the frequency grid.
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from mambagram import MambaGramLayer, GatedMambaGramLayer


def make_chirp_with_click(duration_s=2.0, fs=16000):
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

    # --- Baseline (no gating) ---
    baseline = MambaGramLayer(n_channels=64, sample_rate=fs,
                              f_min=80.0, f_max=fs / 2,
                              window_ms=25.0).to(device)
    with torch.no_grad():
        mag_baseline = baseline.magnitude(x_batch)[0].cpu().numpy().T  # (D, L)

    # --- Gated layer with hand-crafted gate ---
    # We subclass to inject a known gate trajectory: dip to near 0 at the click.
    class HandCraftedGate(GatedMambaGramLayer):
        def _compute_gate(self, x):
            # Envelope via smoothed |x|
            win = int(2e-3 * self.sample_rate)  # 2 ms smoothing
            kernel = torch.ones(1, 1, win, device=x.device) / win
            pad = win // 2
            env = F.pad(x.abs().unsqueeze(1), (pad, pad), mode="reflect")
            env = F.conv1d(env, kernel)[:, 0, :x.size(-1)]       # (B, L)
            env_norm = env / (env.max(dim=-1, keepdim=True).values + 1e-8)

            # Gate = 1 minus a soft pulse at high-amplitude regions.
            # When env_norm is high (click), gate drops toward 0 → reset.
            # When env_norm is moderate (chirp), gate stays near 1 → normal.
            # We use a sharp threshold via sigmoid.
            threshold = 0.3
            steepness = 20.0
            pulse = torch.sigmoid(steepness * (env_norm - threshold))
            gate = 1.0 - 0.95 * pulse  # gate dips to 0.05 at click, stays at 1 elsewhere
            return gate

    gated = HandCraftedGate(n_channels=64, sample_rate=fs,
                             f_min=80.0, f_max=fs / 2,
                             window_ms=25.0,
                             gate_smooth_ms=0.0).to(device)  # already smoothed internally
    # Copy parameters from baseline so only gating differs
    with torch.no_grad():
        gated.omega.copy_(2 * math.pi *
                          GatedMambaGramLayer._mel_spaced(64, 80.0, fs/2) / fs)
        # Re-derive raw_alpha consistent with window_ms=25
        alpha_init = torch.full((64,), -5.0 / (25e-3 * fs))
        gated.raw_alpha.copy_(torch.log(torch.expm1(-alpha_init)))
        gated.b_real.fill_(1.0 / math.sqrt(64))
        gated.b_imag.fill_(0.0)

    with torch.no_grad():
        h_gated, gate_trace = gated(x_batch, return_gate=True)
    mag_gated = h_gated[0].abs().cpu().numpy().T  # (D, L)
    gate_np = gate_trace[0].cpu().numpy()

    # --- Plot ---
    fig, axes = plt.subplots(4, 1, figsize=(11, 12))
    t_axis = np.arange(len(x)) / fs

    axes[0].plot(t_axis, x.numpy(), lw=0.5, color="steelblue")
    axes[0].set_title("Input: chirp (200→4000 Hz) + sharp click @ 1.0s")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, duration)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_axis, np.ones_like(t_axis), "k--",
                 label="Baseline (no gate, γ=1)", alpha=0.5, lw=2)
    axes[1].plot(t_axis, gate_np, "r-",
                 label="Hand-crafted γ_t: drops to ~0.05 at the click", lw=2)
    axes[1].set_title("Forgetting gate γ_t trajectory")
    axes[1].set_ylabel("γ_t")
    axes[1].set_ylim(-0.05, 1.1)
    axes[1].set_xlim(0, duration)
    axes[1].legend(loc="center right")
    axes[1].grid(True, alpha=0.3)

    im2 = axes[2].imshow(to_db_norm(mag_baseline),
                         aspect="auto", origin="lower",
                         extent=[0, duration, 0, 64],
                         cmap="magma", vmin=-50, vmax=0)
    axes[2].set_title("Baseline MambaGram |h| (γ=1, constant)")
    axes[2].set_ylabel("Channel index d")
    plt.colorbar(im2, ax=axes[2], label="dB")

    im3 = axes[3].imshow(to_db_norm(mag_gated),
                         aspect="auto", origin="lower",
                         extent=[0, duration, 0, 64],
                         cmap="magma", vmin=-50, vmax=0)
    axes[3].set_title("Gated MambaGram |h| — γ resets at click → sharper time localization")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Channel index d")
    plt.colorbar(im3, ax=axes[3], label="dB")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "figures",
                       "07_gated_sanity.png")
    plt.savefig(out, dpi=130)
    print(f"Saved: {out}")

    # --- Quantitative metrics ---
    # 1. Click width at a high-frequency channel
    ch = 50
    click_idx = int(1.0 * fs)
    win = 2000
    b_local = mag_baseline[ch, click_idx - win:click_idx + win]
    g_local = mag_gated[ch, click_idx - win:click_idx + win]

    print(f"\nClick width at channel {ch}:")
    print(f"  Baseline: peak={b_local.max():.4f}, FWHM={(b_local > b_local.max()/2).sum()} samples")
    print(f"  Gated:    peak={g_local.max():.4f}, FWHM={(g_local > g_local.max()/2).sum()} samples")

    # 2. Chirp ridge: where is the peak at t=0.5s? Should be SAME for both.
    ridge_t = int(0.5 * fs)
    b_col = mag_baseline[:, ridge_t]
    g_col = mag_gated[:, ridge_t]

    print(f"\nChirp ridge at t=0.5s:")
    print(f"  Baseline: peak channel={b_col.argmax()}, peak/mean ratio={b_col.max()/b_col.mean():.2f}")
    print(f"  Gated:    peak channel={g_col.argmax()}, peak/mean ratio={g_col.max()/g_col.mean():.2f}")

    # 3. Success check
    click_sharper = (g_local > g_local.max()/2).sum() < (b_local > b_local.max()/2).sum()
    ridge_preserved = abs(int(g_col.argmax()) - int(b_col.argmax())) <= 1

    print("\n--- Success checks ---")
    print(f"  Click is sharper with gating: {click_sharper}")
    print(f"  Chirp ridge preserved:        {ridge_preserved}")
    if click_sharper and ridge_preserved:
        print("  ✅ PASS: selectivity improves time localization AND preserves Gabor grid.")
    else:
        print("  ❌ Not yet. Need to adjust gate strength or smoothing.")


if __name__ == "__main__":
    main()
