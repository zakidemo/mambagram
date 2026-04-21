"""
Step 12C: Train MambaGram and Mel-spectrogram frontends on ESC-50
environmental sound classification (50 classes, balanced).

Standard reporting metric: top-1 accuracy on fold 1 test set.
Published simple-CNN baselines hit ~50-65% on ESC-50 without
pretraining; state-of-the-art pretrained models hit 90%+.

Outputs:
  results/03_esc50/history_<frontend>.npz
  results/03_esc50/report_<frontend>.txt
  figures/12_esc50_training_curves.png
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report

from mambagram import AudioClassifier, ESC50Dataset


PROJECT_ROOT = Path(__file__).parent.parent
ESC50_ROOT = PROJECT_ROOT / "data" / "esc50" / "ESC-50-master"
CACHE_DIR  = PROJECT_ROOT / "data" / "esc50" / "cache"
RESULTS    = PROJECT_ROOT / "results" / "03_esc50"
FIG_PATH   = PROJECT_ROOT / "figures" / "12_esc50_training_curves.png"


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    ys, preds, loss_sum, n = [], [], 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * y.size(0)
        preds.append(logits.argmax(-1).cpu().numpy())
        ys.append(y.cpu().numpy())
        n += y.size(0)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    acc = float((y_pred == y_true).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro",
                                labels=list(range(50)), zero_division=0))
    return {
        "loss": loss_sum / n,
        "accuracy": acc,
        "macro_f1": macro_f1,
    }, y_true, y_pred


def train_one(frontend: str, device: str, n_epochs: int = 40,
              lr: float = 3e-4, batch_size: int = 32,
              seed: int = 42) -> dict:
    print(f"\n{'='*70}")
    print(f"Training frontend = {frontend}")
    print(f"{'='*70}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = ESC50Dataset(ESC50_ROOT, split="train", test_fold=1,
                             duration_s=5.0, cache_dir=CACHE_DIR,
                             precompute=False)
    test_ds  = ESC50Dataset(ESC50_ROOT, split="test",  test_fold=1,
                             duration_s=5.0, cache_dir=CACHE_DIR,
                             precompute=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True, num_workers=2,
                               pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size,
                               shuffle=False, num_workers=2,
                               pin_memory=True)

    print(f"Train: {len(train_ds)} | Test: {len(test_ds)} | 50 classes")

    # Model
    model = AudioClassifier(frontend=frontend, n_classes=50,
                             sample_rate=16000, n_features=64).to(device)
    n_params = model.count_parameters()
    print(f"Model: {n_params:,} parameters")

    # Parameter groups: frontend at higher LR for MambaGram
    frontend_param_names = {n for n, _ in model.named_parameters()
                             if n.startswith("frontend.")}
    frontend_params = [p for n, p in model.named_parameters()
                        if n in frontend_param_names]
    head_params = [p for n, p in model.named_parameters()
                   if n not in frontend_param_names]
    frontend_lr = lr * 3.0 if frontend == "mambagram" else lr
    optimizer = torch.optim.Adam([
        {"params": frontend_params, "lr": frontend_lr, "weight_decay": 0.0},
        {"params": head_params,     "lr": lr,           "weight_decay": 1e-4},
    ])
    print(f"Frontend LR: {frontend_lr}, Head LR: {lr}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )
    criterion = nn.CrossEntropyLoss()

    history = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [], "macro_f1": [],
        "epoch_time": [],
    }
    best_acc = -1.0
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        t0 = time.time()
        total, correct, loss_sum = 0, 0, 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.size(0)

        scheduler.step()
        train_loss = loss_sum / total
        train_acc  = correct / total

        test_m, y_true, y_pred = evaluate(model, test_loader, device, criterion)
        dt = time.time() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_m["loss"])
        history["test_acc"].append(test_m["accuracy"])
        history["macro_f1"].append(test_m["macro_f1"])
        history["epoch_time"].append(dt)

        print(f"Epoch {epoch:02d}/{n_epochs} | "
              f"tr_loss={train_loss:.3f} tr_acc={train_acc:.3f} | "
              f"te_loss={test_m['loss']:.3f} "
              f"te_acc={test_m['accuracy']:.3f} "
              f"macF1={test_m['macro_f1']:.3f} | "
              f"{dt:.1f}s")

        if test_m["accuracy"] > best_acc:
            best_acc = test_m["accuracy"]
            best_state = {
                "epoch": epoch,
                "metrics": test_m,
                "y_true": y_true.copy(),
                "y_pred": y_pred.copy(),
            }

    best_m = best_state["metrics"]
    print(f"\nBest top-1 accuracy: {best_m['accuracy']:.4f} (epoch {best_state['epoch']})")
    print(f"  Macro F1: {best_m['macro_f1']:.4f}")

    return {"history": history, "best": best_state, "n_params": n_params}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    frontends = ["mel", "mambagram"]
    results = {}

    for front in frontends:
        results[front] = train_one(front, device, n_epochs=40)

        hist = results[front]["history"]
        np.savez(
            RESULTS / f"history_{front}.npz",
            **{k: np.array(v) for k, v in hist.items()},
        )
        with open(RESULTS / f"report_{front}.txt", "w") as f:
            b = results[front]["best"]
            f.write(f"Frontend: {front}\n")
            f.write(f"Best top-1 accuracy: {b['metrics']['accuracy']:.4f} "
                    f"(epoch {b['epoch']})\n")
            f.write(f"Macro F1: {b['metrics']['macro_f1']:.4f}\n")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"mel": "#1f77b4", "mambagram": "#ff7f0e"}
    labels = {"mel": "Mel-spectrogram", "mambagram": "MambaGram"}

    for front in frontends:
        h = results[front]["history"]
        axes[0].plot(h["epoch"], h["test_loss"], "-o", ms=3,
                     color=colors[front], label=labels[front])
        axes[1].plot(h["epoch"], h["test_acc"], "-o", ms=3,
                     color=colors[front], label=labels[front])

    axes[0].set_title("Test loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-entropy")
    axes[0].grid(alpha=0.3); axes[0].legend()

    axes[1].set_title("Test top-1 accuracy (ESC-50, 50 classes)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1); axes[1].grid(alpha=0.3); axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=130)
    print(f"\nSaved: {FIG_PATH}")

    # Summary
    print("\n" + "="*70)
    print("ESC-50 final summary (best checkpoint per frontend):")
    print("="*70)
    for front in frontends:
        b = results[front]["best"]["metrics"]
        n = results[front]["n_params"]
        print(f"  {labels[front]:30s} | params={n:7,} | "
              f"top1_acc={b['accuracy']:.4f}  macF1={b['macro_f1']:.4f}")


if __name__ == "__main__":
    main()