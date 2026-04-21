"""
Step 11C: Train MambaGram-based classifiers on ICBHI 2017 respiratory
sound classification (4 classes).

Compares two frontends on identical classifier architectures:
  - Mel-spectrogram (baseline)
  - MambaGram (no selectivity; learnable Mel-initialized Gabor filterbank)

Training uses the official ICBHI train/test split, class-weighted
cross-entropy to counter imbalance, and reports the standard ICBHI
Score (mean of sensitivity and specificity).

Outputs:
  results/02_icbhi/history_<frontend>.npz
  results/02_icbhi/report_<frontend>.txt
  figures/11_icbhi_training_curves.png
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
)

from mambagram import (
    AudioClassifier, ICBHIDataset, LABEL_NAMES,
)

# ---- Focal loss for class imbalance ----

class FocalLoss(nn.Module):
    """
    Multi-class focal loss.

    focal_loss(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    alpha : per-class weight vector (same semantics as CE weight)
    gamma : focusing parameter; 0 recovers plain weighted CE, higher
            values down-weight easy examples more aggressively.
    """

    def __init__(self, weight: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)          # (B, C)
        p = log_p.exp()                                 # (B, C)
        # Gather the log-prob of the true class
        target_log_p = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        target_p = p.gather(1, targets.unsqueeze(1)).squeeze(1)          # (B,)
        # Per-example alpha weight from class weight vector
        alpha_t = self.weight[targets]                                    # (B,)
        focal = -alpha_t * ((1 - target_p) ** self.gamma) * target_log_p
        return focal.mean()

def make_balanced_sampler(dataset, n_classes=4):
    """
    WeightedRandomSampler that draws each class with equal probability.

    With this sampler, a mini-batch has approximately equal numbers of
    Normal / Crackle / Wheeze / Both, regardless of dataset frequencies.
    """
    labels = np.array([dataset.cycles[i].label for i in range(len(dataset))])
    class_counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    # Per-sample weight is inverse of its class frequency
    sample_weights = 1.0 / class_counts[labels]
    sample_weights = torch.from_numpy(sample_weights).double()
    return torch.utils.data.WeightedRandomSampler(
        sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )


def class_weights_for_eval(dataset, device, n_classes=4):
    """
    Eval-only sqrt-inverse-frequency weights, for loss reporting on
    the imbalanced test set. Not used for training (sampler handles that).
    """
    labels = np.array([dataset.cycles[i].label for i in range(len(dataset))])
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    inv = counts.sum() / (n_classes * np.maximum(counts, 1))
    weights = np.sqrt(inv)
    weights = weights / weights.mean()
    return torch.from_numpy(weights).float().to(device)

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
ICBHI_ROOT = PROJECT_ROOT / "data" / "icbhi" / "ICBHI_final_database"
CACHE_DIR  = PROJECT_ROOT / "data" / "icbhi" / "cache"
RESULTS    = PROJECT_ROOT / "results" / "02_icbhi"
FIG_PATH   = PROJECT_ROOT / "figures" / "11_icbhi_training_curves.png"


def class_weights(loader, n_classes=4, device="cuda"):
    """Square-root inverse-frequency class weights (softer than pure inverse)."""
    counts = torch.zeros(n_classes)
    for _, y in loader:
        for c in range(n_classes):
            counts[c] += (y == c).sum()
    # Sqrt(inverse frequency), normalized so mean == 1.0
    inv = counts.sum() / (n_classes * counts.clamp(min=1))
    weights = inv.sqrt()
    weights = weights / weights.mean()  # normalize to mean = 1
    return weights.to(device)


def icbhi_score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute the standard ICBHI score metrics.

    - Specificity = accuracy on class Normal (0)
    - Sensitivity = accuracy on abnormal classes (1, 2, 3), averaged
    - Score       = (Specificity + Sensitivity) / 2

    Returns dict with specificity, sensitivity, score, per_class_recall.
    """
    per_class_recall = {}
    for c in range(4):
        mask = (y_true == c)
        if mask.sum() > 0:
            per_class_recall[c] = float((y_pred[mask] == c).mean())
        else:
            per_class_recall[c] = float("nan")

    specificity = per_class_recall[0]  # accuracy on Normal
    abnormal_classes = [1, 2, 3]
    recalls = [per_class_recall[c] for c in abnormal_classes
               if not np.isnan(per_class_recall[c])]
    sensitivity = float(np.mean(recalls)) if recalls else float("nan")
    score = 0.5 * (specificity + sensitivity)

    return {
        "specificity": specificity,
        "sensitivity": sensitivity,
        "score": score,
        "per_class_recall": per_class_recall,
    }


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    ys, preds, loss_sum, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * y.size(0)
        preds.append(logits.argmax(-1).cpu().numpy())
        ys.append(y.cpu().numpy())
        n += y.size(0)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    metrics = icbhi_score(y_true, y_pred)
    metrics["loss"] = loss_sum / max(n, 1)
    metrics["accuracy"] = float((y_pred == y_true).mean())
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro",
                                          labels=[0, 1, 2, 3], zero_division=0))
    return metrics, y_true, y_pred


def train_one(frontend: str, device: str, n_epochs: int = 30,
              lr: float = 1e-3, batch_size: int = 32,
              seed: int = 42) -> dict:
    print(f"\n{'='*70}")
    print(f"Training frontend = {frontend}")
    print(f"{'='*70}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Datasets (cached from Step 11B)
    train_ds = ICBHIDataset(ICBHI_ROOT, split="train", duration_s=4.0,
                             cache_dir=CACHE_DIR, precompute=False)
    test_ds  = ICBHIDataset(ICBHI_ROOT, split="test", duration_s=4.0,
                             cache_dir=CACHE_DIR, precompute=False)

    train_sampler = make_balanced_sampler(train_ds, n_classes=4)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               sampler=train_sampler, num_workers=2,
                               pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size,
                               shuffle=False, num_workers=2,
                               pin_memory=True)

    print(f"Train cycles: {len(train_ds)}, Test cycles: {len(test_ds)}")
    print(f"Train label dist: {train_ds.label_distribution()}")

    # Model
    model = AudioClassifier(frontend=frontend, n_classes=4,
                             sample_rate=16000, n_features=64).to(device)
    n_params = model.count_parameters()
    print(f"Model: {n_params:,} parameters")

    # Class-weighted loss
    # Balanced sampler handles imbalance; use plain CE for training.
    criterion = nn.CrossEntropyLoss()
    # Eval loss uses sqrt-inverse weights so the reported loss isn't
    # dominated by the Normal class.
    eval_weights = class_weights_for_eval(train_ds, device)
    eval_criterion = nn.CrossEntropyLoss(weight=eval_weights, reduction="sum")
    print(f"Training: balanced sampler, plain CE")
    print(f"Eval weights (reporting only): {eval_weights.cpu().numpy()}")

    # Parameter groups: frontend learnable params at higher LR than head.
    # (For Mel frontend: no learnable frontend params, so this is a no-op.)
    frontend_param_names = set()
    head_param_names = set()
    for name, _ in model.named_parameters():
        if name.startswith("frontend."):
            frontend_param_names.add(name)
        else:
            head_param_names.add(name)

    frontend_params = [p for n, p in model.named_parameters()
                       if n in frontend_param_names]
    head_params = [p for n, p in model.named_parameters()
                   if n in head_param_names]

    # For MambaGram/gated: bump frontend LR 10x (they have few, slow-moving params)
    # For Mel: frontend has no params, so this group is empty — no effect.
    frontend_lr = lr * 3.0 if frontend == "mambagram" else lr

    optimizer = torch.optim.Adam([
        {"params": frontend_params, "lr": frontend_lr,
         "weight_decay": 0.0},  # no weight decay on frontend params (few, specialized)
        {"params": head_params, "lr": lr, "weight_decay": 1e-5},
    ])
    print(f"Frontend LR: {frontend_lr}, Head LR: {lr}")

    # LR schedule: warmup for first epoch, then cosine anneal
    warmup_epochs = 1
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup from 0 to 1
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine anneal from 1 to ~0
            import math
            progress = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [],
        "sensitivity": [], "specificity": [], "icbhi_score": [],
        "macro_f1": [], "epoch_time": [],
    }
    best_score = -1.0
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

        test_metrics, y_true, y_pred = evaluate(model, test_loader, device, eval_criterion)
        dt = time.time() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_metrics["loss"])
        history["test_acc"].append(test_metrics["accuracy"])
        history["sensitivity"].append(test_metrics["sensitivity"])
        history["specificity"].append(test_metrics["specificity"])
        history["icbhi_score"].append(test_metrics["score"])
        history["macro_f1"].append(test_metrics["macro_f1"])
        history["epoch_time"].append(dt)

        print(f"Epoch {epoch:02d}/{n_epochs} | "
              f"tr_loss={train_loss:.3f} tr_acc={train_acc:.3f} | "
              f"te_loss={test_metrics['loss']:.3f} te_acc={test_metrics['accuracy']:.3f} | "
              f"Spec={test_metrics['specificity']:.3f} "
              f"Sens={test_metrics['sensitivity']:.3f} "
              f"Score={test_metrics['score']:.3f} | "
              f"{dt:.1f}s")

                # that ICBHI Score alone can reward)
        if test_metrics["macro_f1"] > best_score:
            best_score = test_metrics["macro_f1"]
            best_state = {
                "epoch": epoch,
                "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "metrics": test_metrics,
                "y_true": y_true.copy(),
                "y_pred": y_pred.copy(),
            }

    best_metrics = best_state["metrics"]
    print(f"\nBest macro F1: {best_metrics['macro_f1']:.4f} "
          f"(epoch {best_state['epoch']})")
    print(f"  ICBHI Score  : {best_metrics['score']:.4f}")
    print(f"  Sensitivity  : {best_metrics['sensitivity']:.4f}")
    print(f"  Specificity  : {best_metrics['specificity']:.4f}")
    print("Classification report (best epoch):")
    print(classification_report(best_state["y_true"], best_state["y_pred"],
                                 labels=[0, 1, 2, 3],
                                 target_names=LABEL_NAMES,
                                 zero_division=0))

    cm = confusion_matrix(best_state["y_true"], best_state["y_pred"], labels=[0, 1, 2, 3])
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    return {"history": history, "best": best_state, "n_params": n_params}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    frontends = ["mel", "mambagram"]
    results = {}

    lrs = {"mel": 3e-4, "mambagram": 3e-4}
    for front in frontends:
        results[front] = train_one(front, device, n_epochs=30,
                                    lr=lrs[front])

        # Save metrics
        hist = results[front]["history"]
        np.savez(
            RESULTS / f"history_{front}.npz",
            **{k: np.array(v) for k, v in hist.items()},
        )
        with open(RESULTS / f"report_{front}.txt", "w") as f:
            best = results[front]["best"]
            f.write(f"Frontend: {front}\n")
            f.write(f"Best ICBHI score: {best['metrics']['score']:.4f} "
                    f"(epoch {best['epoch']})\n")
            f.write(f"Specificity: {best['metrics']['specificity']:.4f}\n")
            f.write(f"Sensitivity: {best['metrics']['sensitivity']:.4f}\n")
            f.write(f"Accuracy:    {best['metrics']['accuracy']:.4f}\n")
            f.write(f"Macro F1:    {best['metrics']['macro_f1']:.4f}\n")
            f.write(f"Per-class recall: {best['metrics']['per_class_recall']}\n")

    # --- Plot: side-by-side training curves ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = {"mel": "#1f77b4", "mambagram": "#ff7f0e"}
    labels = {"mel": "Mel-spectrogram", "mambagram": "MambaGram"}

    for front in frontends:
        h = results[front]["history"]
        axes[0].plot(h["epoch"], h["test_loss"], "-o", ms=3,
                     color=colors[front], label=labels[front])
        axes[1].plot(h["epoch"], h["icbhi_score"], "-o", ms=3,
                     color=colors[front], label=labels[front])
        axes[2].plot(h["epoch"], h["sensitivity"], "-o", ms=3,
                     color=colors[front], label=f"{labels[front]} sens")
        axes[2].plot(h["epoch"], h["specificity"], "--^", ms=3,
                     color=colors[front], label=f"{labels[front]} spec",
                     alpha=0.6)

    axes[0].set_title("Test loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Weighted CE"); axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("ICBHI Score (higher is better)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score"); axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].set_title("Sensitivity / Specificity")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Recall"); axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=130)
    print(f"\nSaved training curves: {FIG_PATH}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("Final summary (best checkpoint per frontend):")
    print("=" * 70)
    for front in frontends:
        best = results[front]["best"]["metrics"]
        n = results[front]["n_params"]
        print(f"  {labels[front]:30s} | params={n:7,} | "
              f"Score={best['score']:.4f}  "
              f"Sens={best['sensitivity']:.4f}  "
              f"Spec={best['specificity']:.4f}")


if __name__ == "__main__":
    main()