"""
Step 10: Train MambaGram-based classifiers on synthetic 4-class audio.

Compares three frontends on the same synthetic task:
  - Mel-spectrogram (baseline)
  - MambaGram (no selectivity)
  - GatedMambaGram (with selectivity)

All three use identical classifier heads. Differences in accuracy
reflect differences in the TF representation only.

Outputs:
  results/01_synthetic/history_<frontend>.npz    — per-epoch metrics
  figures/10_synthetic_training_curves.png       — comparison plot
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from mambagram import AudioClassifier, SyntheticAudioDataset


def make_loaders(n_train=4000, n_val=800, batch_size=32, seed=0):
    train_ds = SyntheticAudioDataset(n_samples=n_train, seed=seed)
    val_ds = SyntheticAudioDataset(n_samples=n_val, seed=seed + 999)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader


def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += criterion(logits, y).item()
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


def train_one(frontend: str, device: str, n_epochs: int = 10,
              lr: float = 3e-3, seed: int = 0) -> dict:
    """Train one model and return per-epoch history."""
    torch.manual_seed(seed)
    print(f"\n{'='*60}")
    print(f"Training frontend = {frontend}")
    print(f"{'='*60}")

    train_loader, val_loader = make_loaders(seed=seed)
    model = AudioClassifier(frontend=frontend, n_classes=4).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "epoch_time": [],
    }

    for epoch in range(1, n_epochs + 1):
        model.train()
        t0 = time.time()
        total, correct, loss_sum = 0, 0, 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.size(0)

        scheduler.step()
        train_loss = loss_sum / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, device)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(dt)

        print(f"Epoch {epoch:02d}/{n_epochs} | "
              f"train loss={train_loss:.3f} acc={train_acc:.3f} | "
              f"val loss={val_loss:.3f} acc={val_acc:.3f} | "
              f"{dt:.1f}s")

    return history


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results_dir = Path(__file__).parent.parent / "results" / "01_synthetic"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_path = Path(__file__).parent.parent / "figures" / "10_synthetic_training_curves.png"

    histories = {}
    for frontend in ["mel", "mambagram", "gated"]:
        histories[frontend] = train_one(frontend, device, n_epochs=10)
        np.savez(
            results_dir / f"history_{frontend}.npz",
            **{k: np.array(v) for k, v in histories[frontend].items()},
        )

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"mel": "#1f77b4", "mambagram": "#ff7f0e", "gated": "#2ca02c"}
    labels = {"mel": "Mel-spectrogram (baseline)",
              "mambagram": "MambaGram (no selectivity)",
              "gated": "GatedMambaGram (selective)"}

    for front, hist in histories.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["val_loss"], "-o",
                     color=colors[front], label=labels[front])
        axes[1].plot(epochs, hist["val_acc"], "-o",
                     color=colors[front], label=labels[front])

    axes[0].set_title("Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.2, 1.02)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(fig_path, dpi=130)
    print(f"\nSaved training curves: {fig_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Final validation accuracy after 10 epochs:")
    print("=" * 60)
    for front, hist in histories.items():
        final_acc = hist["val_acc"][-1]
        best_acc = max(hist["val_acc"])
        avg_time = np.mean(hist["epoch_time"])
        print(f"  {labels[front]:40s} final={final_acc:.3f}  best={best_acc:.3f}  "
              f"({avg_time:.1f}s/epoch)")


if __name__ == "__main__":
    main()