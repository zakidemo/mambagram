"""
Diagnostic A: Track how much MambaGram frontend parameters move during
training. Informs whether LR is too low (parameters stuck) or too high
(parameters thrashing).

Logs the L2 distance of each parameter group from its initial value at
fixed intervals during a short training run.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mambagram import AudioClassifier, ICBHIDataset


PROJECT_ROOT = Path(__file__).parent.parent
ICBHI_ROOT = PROJECT_ROOT / "data" / "icbhi" / "ICBHI_final_database"
CACHE_DIR  = PROJECT_ROOT / "data" / "icbhi" / "cache"


def main():
    device = "cuda"
    torch.manual_seed(42)

    ds = ICBHIDataset(ICBHI_ROOT, split="train", duration_s=4.0,
                      cache_dir=CACHE_DIR, precompute=False)
    loader = DataLoader(ds, batch_size=32, shuffle=True,
                         num_workers=2, pin_memory=True, drop_last=True)

    # Use the same setup as training: frontend LR = 3e-3, head LR = 3e-4
    model = AudioClassifier(frontend="mambagram", n_classes=4).to(device)

    # Snapshot initial values of frontend parameters
    fe = model.frontend.layer
    init = {
        "omega":     fe.omega.detach().clone(),
        "raw_alpha": fe.raw_alpha.detach().clone(),
        "b_real":    fe.b_real.detach().clone(),
        "b_imag":    fe.b_imag.detach().clone(),
    }
    # Compute norms of initial values for relative drift reporting
    init_norms = {k: v.norm().item() for k, v in init.items()}

    frontend_params = [fe.omega, fe.raw_alpha, fe.b_real, fe.b_imag]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("frontend.")]

    # Try two LR regimes
    for label, lr_fe, lr_head in [
        ("10x (the one that collapsed)", 3e-3, 3e-4),
        ("3x (proposed replacement)",    9e-4, 3e-4),
        ("1x (baseline — old run)",      3e-4, 3e-4),
    ]:
        # Reset model to original state
        fe.omega.data.copy_(init["omega"])
        fe.raw_alpha.data.copy_(init["raw_alpha"])
        fe.b_real.data.copy_(init["b_real"])
        fe.b_imag.data.copy_(init["b_imag"])
        # Re-init head (matters less but for clean comparison)
        torch.manual_seed(42)
        model.head.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

        optimizer = torch.optim.Adam([
            {"params": frontend_params, "lr": lr_fe},
            {"params": head_params, "lr": lr_head, "weight_decay": 1e-5},
        ])
        criterion = nn.CrossEntropyLoss()

        print(f"\n=== {label} (frontend LR={lr_fe}, head LR={lr_head}) ===")

        model.train()
        it = iter(loader)
        losses = []
        # Train for 100 batches
        for step in range(100):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        # Measure parameter drift
        drift = {
            "omega":     (fe.omega - init["omega"]).norm().item(),
            "raw_alpha": (fe.raw_alpha - init["raw_alpha"]).norm().item(),
            "b_real":    (fe.b_real - init["b_real"]).norm().item(),
            "b_imag":    (fe.b_imag - init["b_imag"]).norm().item(),
        }
        print(f"Average train loss: {np.mean(losses):.4f}  "
              f"Final train loss: {np.mean(losses[-10:]):.4f}")
        print(f"Parameter drift (L2 distance from init):")
        for k in ["omega", "raw_alpha", "b_real", "b_imag"]:
            rel = drift[k] / max(init_norms[k], 1e-8) * 100
            print(f"  {k:10s}  abs={drift[k]:7.4f}  rel={rel:6.2f}% of init norm")


if __name__ == "__main__":
    main()