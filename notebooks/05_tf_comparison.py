"""
Step 8: Side-by-side comparison of MambaGram vs classical time-frequency
representations on three signal archetypes.

Produces the paper's candidate Figure 1: a 4x3 grid showing how each
representation handles each signal type.

Rows   : STFT, CQT, Wavelet scalogram (CWT), MambaGram
Columns: Linear chirp, Gaussian click, Harmonic stack

The scientific claim: Classical representations excel only on their
"home" signal type (STFT for chirps, CQT for harmonics, scalogram for
transients). MambaGram, with a single fixed parameterization, produces
competitive representations across ALL signal types.
"""
import math
import os
import sys
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Silence ssqueezepy info logs
warnings.filterwarnings("ignore")
os.environ["SSQUEEZEPY_VERBOSE"] = "0"

import librosa
from ssqueezepy import cwt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mambagram import MambaGramLayer


# --- Signal generators (same as Step 6 but deterministic) ---

def make_linear_chirp(duration_s=2.0, fs=16000, f0=200.0, f1=4000.0):
    t = torch.arange(0, int(duration_s * fs)) / fs
    phase = 2 * math.pi * (f0 * t + 0.5 * (f1 - f0) / duration_s * t**2)
    return torch.sin(phase).float()


def make_click(duration_s=2.0, fs=16000, click_time_s=1.0, width_s=0.002, seed=0):
    torch.manual_seed(seed)
    t = torch.arange(0, int(duration_s * fs)) / fs
    click = torch.exp(-0.5 * ((t - click_time_s) / width_s) ** 2)
    noise = 0.01 * torch.randn_like(click)
    return (click + noise).float()


def make_harmonic_stack(duration_s=2.0, fs=16000, f0=220.0, n_harmonics=6):
    t = torch.arange(0, int(duration_s * fs)) / fs
    y = torch.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        if k * f0 >= fs / 2:
            break
        y += (1.0 / k) * torch.sin(2 * math.pi * k * f0 * t)
    y = y / y.abs().max()
    return y.float()


# --- TF representation wrappers ---
# All return a 2D array (freq_bins, time_frames) and the corresponding
# frequency vector in Hz, so we can plot them consistently.

def compute_stft(x_np, fs, n_fft=512, hop=128):
    """Standard STFT magnitude."""
    X = librosa.stft(x_np, n_fft=n_fft, hop_length=hop, window="hann")
    mag = np.abs(X)
    freqs = np.linspace(0, fs / 2, mag.shape[0])
    times = np.arange(mag.shape[1]) * hop / fs
    return mag, freqs, times


def compute_cqt(x_np, fs, hop=128, bins_per_octave=24, n_bins=168, fmin=55.0):
    """Constant-Q Transform magnitude."""
    C = librosa.cqt(x_np, sr=fs, hop_length=hop,
                    fmin=fmin, n_bins=n_bins,
                    bins_per_octave=bins_per_octave)
    mag = np.abs(C)
    freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin,
                                     bins_per_octave=bins_per_octave)
    times = np.arange(mag.shape[1]) * hop / fs
    return mag, freqs, times


def compute_scalogram(x_np, fs, n_scales=128):
    """Continuous wavelet transform magnitude (Morlet-like) via ssqueezepy."""
    # ssqueezepy.cwt returns (Wx, scales). Wx is complex, shape (n_scales, L).
    Wx, scales = cwt(x_np, wavelet="gmw", nv=32)
    mag = np.abs(Wx)
    # Convert scales to approximate frequencies (for Morlet-like wavelet)
    # For GMW wavelet, center frequency ~ 0.25 in normalized units
    center_freq = 0.25
    freqs = center_freq * fs / scales
    times = np.arange(mag.shape[1]) / fs
    return mag, freqs, times


def compute_mambagram(x_np, fs, device, n_channels=64, f_min=80.0,
                      window_ms=25.0):
    """MambaGram magnitude at initialization (no training)."""
    layer = MambaGramLayer(
        n_channels=n_channels, sample_rate=fs,
        f_min=f_min, f_max=fs / 2,
        window_ms=window_ms, init="mel",
    ).to(device)
    x_t = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
    with torch.no_grad():
        mag = layer.magnitude(x_t)[0].cpu().numpy().T  # (D, L)
    freqs = (layer.omega.detach().cpu().numpy() * fs) / (2 * math.pi)
    times = np.arange(mag.shape[1]) / fs
    return mag, freqs, times


# --- Plotting ---

def to_db_norm(mag, eps=1e-8):
    """Normalize to peak, convert to dB."""
    mag = mag / (mag.max() + eps)
    return 20 * np.log10(mag + eps)


def plot_tf(ax, mag_db, freqs, times, title, ylim=(80, 8000), vmin=-60, vmax=0):
    """Plot a TF representation with log-y frequency axis."""
    im = ax.pcolormesh(
        times, freqs, mag_db,
        cmap="magma", vmin=vmin, vmax=vmax, shading="auto",
    )
    ax.set_yscale("log")
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.tick_params(labelsize=8)
    return im


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = 16000
    duration = 2.0

    # --- Generate signals ---
    signals = {
        "Chirp (200→4000 Hz)": make_linear_chirp(duration, fs),
        "Click @ 1.0 s": make_click(duration, fs),
        "Harmonic stack (f₀=220 Hz)": make_harmonic_stack(duration, fs),
    }

    print(f"Device: {device}, sample rate: {fs} Hz, duration: {duration}s\n")

    # --- Compute all four representations for all three signals ---
    print("Computing TF representations...")
    results = {}  # {signal_name: {method: (mag_db, freqs, times)}}

    for sig_name, x_t in signals.items():
        x_np = x_t.numpy()
        print(f"  [{sig_name}]")

        stft_mag, stft_f, stft_t = compute_stft(x_np, fs)
        print(f"    STFT      : {stft_mag.shape}")

        cqt_mag, cqt_f, cqt_t = compute_cqt(x_np, fs)
        print(f"    CQT       : {cqt_mag.shape}")

        cwt_mag, cwt_f, cwt_t = compute_scalogram(x_np, fs)
        # Subsample CWT in time for plotting speed (it's sample-rate dense)
        cwt_sub_t = cwt_t[::64]
        cwt_sub_mag = cwt_mag[:, ::64]
        print(f"    Scalogram : {cwt_mag.shape} -> subsampled to {cwt_sub_mag.shape}")

        mg_mag, mg_f, mg_t = compute_mambagram(x_np, fs, device)
        print(f"    MambaGram : {mg_mag.shape}")

        results[sig_name] = {
            "STFT": (to_db_norm(stft_mag), stft_f, stft_t),
            "CQT": (to_db_norm(cqt_mag), cqt_f, cqt_t),
            "Scalogram (CWT)": (to_db_norm(cwt_sub_mag), cwt_f, cwt_sub_t),
            "MambaGram": (to_db_norm(mg_mag), mg_f, mg_t),
        }

    # --- Build the 4x3 figure ---
    methods = ["STFT", "CQT", "Scalogram (CWT)", "MambaGram"]
    sig_names = list(signals.keys())

    fig = plt.figure(figsize=(13, 11))
    gs = gridspec.GridSpec(len(methods), len(sig_names),
                           wspace=0.28, hspace=0.45)

    for i, method in enumerate(methods):
        for j, sig_name in enumerate(sig_names):
            ax = fig.add_subplot(gs[i, j])
            mag_db, freqs, times = results[sig_name][method]

            # For methods with y-axis in channel index (not used here — all use Hz),
            # we treat them uniformly.
            title = f"{method} — {sig_name}" if i == 0 else method
            im = plot_tf(ax, mag_db, freqs, times, title=title)

            if j == 0:
                ax.set_ylabel(f"{method}\nFreq (Hz)", fontsize=9)
            else:
                ax.set_ylabel("Freq (Hz)", fontsize=8)

            # Only label x-axis on bottom row
            if i < len(methods) - 1:
                ax.set_xlabel("")

    # Global colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="dB (normalized)")

    fig.suptitle(
        "Time-Frequency Representations Compared on Three Signal Archetypes\n"
        "(MambaGram: untrained, Mel-initialized, D=64 channels)",
        fontsize=12, y=0.995,
    )

    out = os.path.join(os.path.dirname(__file__), "..", "figures",
                       "05_tf_comparison.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}")

    print("\n--- Interpretation ---")
    print("  Each row = one TF method applied to all three signals.")
    print("  Look for:")
    print("    STFT column 1 (chirp): clean diagonal ridge ← STFT's home turf")
    print("    CQT  column 3 (harmonic): evenly spaced horizontal bands ← CQT's home turf")
    print("    CWT  column 2 (click): narrow vertical stripe ← scalogram's home turf")
    print("    MambaGram (row 4): should show ALL THREE behaviors, on the same parameters")


if __name__ == "__main__":
    main()
