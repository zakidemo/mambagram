"""
Benchmark MambaGram FFT path memory and speed at various sequence
lengths and batch sizes. Used to choose ICBHI training parameters.
"""
import time

import torch

from mambagram import MambaGramLayer


def main():
    device = "cuda"
    layer = MambaGramLayer(n_channels=64, sample_rate=16000).to(device)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total VRAM: {total_mem:.2f} GB")
    print()
    print(f"{'duration':>10s} {'batch':>6s} {'time_ms':>10s} {'peak_GB':>10s}")
    print("-" * 42)

    for L_seconds in [1, 2, 4, 8]:
        L = int(L_seconds * 16000)
        for batch in [1, 8, 16, 32]:
            try:
                x = torch.randn(batch, L, device=device)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                with torch.no_grad():
                    h = layer(x, method="fft")
                torch.cuda.synchronize()
                dt_ms = (time.time() - t0) * 1000
                peak_gb = torch.cuda.max_memory_allocated() / 1e9

                print(f"{L_seconds:>9d}s {batch:>6d} "
                      f"{dt_ms:>10.1f} {peak_gb:>10.2f}")

                del x, h
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"{L_seconds:>9d}s {batch:>6d} {'OOM':>10s} {'--':>10s}")
                torch.cuda.empty_cache()
                # Skip larger batches at this L
                break


if __name__ == "__main__":
    main()