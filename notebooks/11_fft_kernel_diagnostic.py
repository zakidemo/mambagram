"""
Diagnostic B: Does the extended FFT kernel (Fix 2) change the MambaGram
output meaningfully, or is it wasted compute?

We compare two versions of the layer:
  - Short kernel: K chosen so |a|^K = 1e-4  (~40 ms effective)
  - Full kernel: K = signal length       (~ 4 seconds effective)

On a real ICBHI clip, we measure:
  1. Maximum absolute difference between the two outputs
  2. Percent of energy captured by short vs full kernel
  3. Speed difference
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from mambagram import ICBHIDataset, MambaGramLayer


PROJECT_ROOT = Path(__file__).parent.parent
ICBHI_ROOT = PROJECT_ROOT / "data" / "icbhi" / "ICBHI_final_database"
CACHE_DIR  = PROJECT_ROOT / "data" / "icbhi" / "cache"


# Monkey-patch the layer to expose kernel-length control
class MambaGramLayerWithK(MambaGramLayer):
    """Adds a runtime K override for the FFT kernel length."""
    def __init__(self, *args, k_mode="full", **kwargs):
        super().__init__(*args, **kwargs)
        self.k_mode = k_mode

    def _forward_fft(self, x):
        import math as _math
        # Keep the original logic but override K based on k_mode
        batch, length = x.shape
        D = self.n_channels
        device = x.device
        dtype = x.dtype

        alpha = self._get_alpha().to(device=device, dtype=dtype)
        omega = self.omega.to(device=device, dtype=dtype)
        b = torch.complex(self.b_real, self.b_imag).to(device=device)

        if self.k_mode == "full":
            K = length
        elif self.k_mode == "short":
            threshold = 1e-4
            alpha_min = alpha.min().clamp(max=-1e-6)
            K_needed = int(_math.ceil(_math.log(threshold) / alpha_min.item()))
            K = min(K_needed, length)
        else:
            raise ValueError(self.k_mode)

        tau = torch.arange(K, device=device, dtype=dtype)
        log_a = torch.complex(alpha, omega)
        exponent = log_a[:, None] * tau[None, :]
        g = b[:, None] * torch.exp(exponent)

        n_fft = length + K - 1
        n_fft = 1 << (n_fft - 1).bit_length()

        x_c = x.to(torch.complex64)
        X = torch.fft.fft(x_c, n=n_fft)
        G = torch.fft.fft(g, n=n_fft)
        Y = X[:, None, :] * G[None, :, :]
        H = torch.fft.ifft(Y, n=n_fft)[:, :, :length]
        H = H.transpose(1, 2).contiguous()
        return H


def bench(layer, x, n=5):
    for _ in range(2):  # warmup
        _ = layer(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        y = layer(x)
    torch.cuda.synchronize()
    return (time.time() - t0) / n, y


def main():
    device = "cuda"
    torch.manual_seed(0)

    ds = ICBHIDataset(ICBHI_ROOT, split="train", duration_s=4.0,
                      cache_dir=CACHE_DIR, precompute=False)

    # Build short-kernel and full-kernel layers with identical parameters
    full = MambaGramLayerWithK(n_channels=64, sample_rate=16000,
                                k_mode="full").to(device)
    short = MambaGramLayerWithK(n_channels=64, sample_rate=16000,
                                 k_mode="short").to(device)
    # Copy parameters from full to short so we're comparing only K
    with torch.no_grad():
        short.omega.copy_(full.omega)
        short.raw_alpha.copy_(full.raw_alpha)
        short.b_real.copy_(full.b_real)
        short.b_imag.copy_(full.b_imag)

    # Compute the actual K values being used
    alpha = full._get_alpha()
    alpha_min = alpha.min().item()
    K_short = int(math.ceil(math.log(1e-4) / alpha_min))
    K_full = int(4.0 * 16000)
    print(f"Short K (threshold=1e-4): {K_short} samples ({K_short/16000*1000:.1f} ms)")
    print(f"Full K (signal length):   {K_full} samples ({K_full/16000*1000:.1f} ms)")
    print(f"Ratio: full is {K_full/K_short:.1f}x longer\n")

    # Take 5 real ICBHI clips, run both versions, compare
    abs_diffs, rel_diffs = [], []
    for idx in [0, 100, 500, 1000, 2000]:
        x, y = ds[idx]
        x_batch = x.unsqueeze(0).to(device)
        with torch.no_grad():
            h_full = full(x_batch)
            h_short = short(x_batch)
        diff = (h_full - h_short).abs()
        abs_diff = diff.max().item()
        # Relative: how much of the signal is being discarded?
        total_energy_full = (h_full.abs() ** 2).sum().item()
        total_energy_short = (h_short.abs() ** 2).sum().item()
        rel_energy_ratio = total_energy_short / max(total_energy_full, 1e-12)
        abs_diffs.append(abs_diff)
        rel_diffs.append(rel_energy_ratio)
        print(f"Clip {idx}: label={y}, "
              f"max_abs_diff={abs_diff:.4f}, "
              f"short_energy/full_energy={rel_energy_ratio:.4f}")

    print(f"\nAverage relative energy captured by short kernel: "
          f"{sum(rel_diffs)/len(rel_diffs):.4f}")
    print(f"Interpretation:")
    print(f"  ~1.00 = short kernel captures essentially all energy; Fix 2 is wasted")
    print(f"  <0.90 = short kernel discards >10% of energy; Fix 2 meaningful")

    # Speed comparison
    print("\n--- Speed benchmark (batch=32) ---")
    x_batch = torch.randn(32, 64000, device=device)
    t_full, _ = bench(full, x_batch)
    t_short, _ = bench(short, x_batch)
    print(f"Full kernel:  {t_full*1000:7.1f} ms/batch")
    print(f"Short kernel: {t_short*1000:7.1f} ms/batch")
    print(f"Full is {t_full/t_short:.1f}x slower")


if __name__ == "__main__":
    main()