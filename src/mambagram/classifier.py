"""
End-to-end audio classifier: TF frontend + small CNN head.

The frontend can be:
  - MambaGramLayer (baseline)
  - GatedMambaGramLayer (selective)
  - torchaudio.transforms.MelSpectrogram (external baseline)

The head is a small 2D CNN followed by global pooling and a linear
classifier. Kept intentionally shallow so representation quality
dominates the result.
"""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as AT

from mambagram.layers import MambaGramLayer
from mambagram.gated_layer import GatedMambaGramLayer


class MelFrontend(nn.Module):
    """Fixed Mel-spectrogram frontend (baseline)."""

    def __init__(self, sample_rate=16000, n_mels=64, n_fft=512, hop=128):
        super().__init__()
        self.mel = AT.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            power=2.0,
        )
        self.log = AT.AmplitudeToDB(stype="power", top_db=80)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) -> mel: (B, n_mels, T)
        mel = self.mel(x)
        mel_db = self.log(mel)
        # (B, n_mels, T) -> (B, T, n_mels) to match MambaGram layout
        return mel_db.transpose(1, 2)


class MambaGramFrontendWrapper(nn.Module):
    """
    Wraps a MambaGram layer to output a log-magnitude spectrogram.

    Output shape: (B, T, D) — same as MelFrontend.
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L) -> (B, L, D) complex -> magnitude -> log
        h = self.layer(x)
        # Magnitude then log-compression (mimics log-Mel)
        mag = h.abs()
        return torch.log1p(mag)  # log(1 + |h|) for numerical stability


class ShallowCNNHead(nn.Module):
    """
    Small 2D CNN classifier operating on (B, T, D) features.

    Intentionally shallow so the representation does the heavy lifting.
    """

    def __init__(self, n_features: int, n_classes: int,
                 hidden: int = 32):
        super().__init__()
        # Treat the feature dim (D) as the 'channel' dimension of a 2D conv.
        # Input: (B, T, D) -> transpose to (B, D, T) -> treat as 1D conv
        # over time with D input channels. We use 2D conv with (T, 1) kernel
        # for compatibility with 2D pooling.
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, T, D) -> (B, D, T)
        x = feat.transpose(1, 2)
        x = self.conv(x)           # (B, 2*hidden, 1)
        x = x.squeeze(-1)          # (B, 2*hidden)
        return self.fc(x)          # (B, n_classes)


class AudioClassifier(nn.Module):
    """
    Full classifier: frontend + head.

    Parameters
    ----------
    frontend : {'mel', 'mambagram', 'gated'}
    n_classes : int
    sample_rate : int
    n_features : int
        Number of feature channels from the frontend (D for MambaGram, n_mels for Mel).
    """

    def __init__(
        self,
        frontend: Literal["mel", "mambagram", "gated"],
        n_classes: int,
        sample_rate: int = 16000,
        n_features: int = 64,
    ):
        super().__init__()
        self.frontend_name = frontend

        if frontend == "mel":
            self.frontend = MelFrontend(sample_rate=sample_rate, n_mels=n_features)
        elif frontend == "mambagram":
            layer = MambaGramLayer(
                n_channels=n_features, sample_rate=sample_rate,
                f_min=80.0, f_max=sample_rate / 2,
                window_ms=25.0, init="mel",
            )
            self.frontend = MambaGramFrontendWrapper(layer)
        elif frontend == "gated":
            layer = GatedMambaGramLayer(
                n_channels=n_features, sample_rate=sample_rate,
                f_min=80.0, f_max=sample_rate / 2,
                window_ms=25.0, gate_smooth_ms=1.0,
            )
            self.frontend = MambaGramFrontendWrapper(layer)
        else:
            raise ValueError(f"Unknown frontend: {frontend}")

        self.head = ShallowCNNHead(n_features=n_features, n_classes=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.frontend(x)      # (B, T, D)
        logits = self.head(feat)     # (B, n_classes)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)