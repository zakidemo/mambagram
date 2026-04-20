"""
Step 7B: Verify the FFT-based forward pass is (a) correct and (b) much faster
than the recurrent reference.

Correctness check: for the same input, _forward_recurrent and _forward_fft
should produce nearly identical outputs (up to floating-point tolerance).

Speed check: we time both methods on a 2-second clip and report the speedup.
"""
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mambagram import MambaGramLayer


def make_linear_chirp(duration_s=2.0, fs=16000, f0=200.0, f1=4000.0):
    t = torch.arange(0, int(duration_s * fs)) / fs
    phase = 2 * math.pi * (f0 * t + 0.5 * (f1 - f0) / duration_s * t**2)
    return torch.sin(phase).float()


def bench(fn, n_warmup=2, n_runs=5):
    """Run fn() a few times and return the median elapsed time in seconds."""
    # Warmup
    for _ in range(n_warmup):
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    times.sort()
    return times[len(times) // 2]  # median


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = 16000
    duration = 2.0

    x = make_linear_chirp(duration, fs).unsqueeze(0).to(device)  # (1, L)
    print(f"Input: batch=1, length={x.shape[1]}, device={device}\n")

    layer = MambaGramLayer(
        n_channels=64, sample_rate=fs,
        f_min=80.0, f_max=fs / 2,
        window_ms=25.0, init="mel",
    ).to(device)

    # --- Correctness check ---
    print("--- Correctness: recurrent vs FFT ---")
    with torch.no_grad():
        H_recurrent = layer(x, method="recurrent")
        H_fft = layer(x, method="fft")

    abs_diff = (H_recurrent - H_fft).abs()
    print(f"Output shapes match: {H_recurrent.shape == H_fft.shape}")
    print(f"Max absolute difference:  {abs_diff.max().item():.2e}")
    print(f"Mean absolute difference: {abs_diff.mean().item():.2e}")

    # Only compute relative diff where the reference output is meaningfully large
    # (otherwise divide-by-almost-zero creates misleading numbers)
    mask = H_recurrent.abs() > 0.01 * H_recurrent.abs().max()
    if mask.any():
        rel_diff = abs_diff[mask] / H_recurrent.abs()[mask]
        print(f"Max rel diff (on significant values): {rel_diff.max().item():.2e}")
        print(f"Mean rel diff (on significant values): {rel_diff.mean().item():.2e}")
    else:
        print("Warning: no significant values in recurrent output")

    if abs_diff.max().item() < 1e-3:
        print("✅ FFT path matches recurrent path within floating-point tolerance.\n")
    else:
        print("❌ Mismatch too large! Investigate before trusting FFT path.\n")

    # --- Speed check ---
    print("--- Speed benchmark (median of 5 runs after 2 warmup) ---")

    t_rec = bench(lambda: layer(x, method="recurrent"))
    print(f"Recurrent (Python loop): {t_rec*1000:7.1f} ms")

    t_fft = bench(lambda: layer(x, method="fft"))
    print(f"FFT (parallel scan):     {t_fft*1000:7.1f} ms")

    print(f"\nSpeedup: {t_rec / t_fft:.1f}x")

    # --- Batch size scaling ---
    print("\n--- Batch size scaling (FFT method only) ---")
    for B in [1, 8, 32, 64]:
        xb = x.repeat(B, 1)
        t = bench(lambda: layer(xb, method="fft"))
        print(f"  batch={B:3d}: {t*1000:7.1f} ms  ({t*1000/B:.2f} ms/clip)")


if __name__ == "__main__":
    main()
