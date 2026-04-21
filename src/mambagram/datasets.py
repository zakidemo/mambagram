"""
Synthetic audio datasets for validating the training pipeline.

Provides a 4-class dataset:
  0 = pure chirp (linearly swept sinusoid)
  1 = chirp + superimposed click
  2 = harmonic tone (sum of harmonics)
  3 = white noise

Each sample is a 1 s clip @ 16 kHz. Labels are deterministic given the
random seed, so train/val/test splits are reproducible.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticAudioDataset(Dataset):
    """
    On-the-fly synthetic audio classification dataset.

    Parameters
    ----------
    n_samples : int
        Total number of samples to generate in this split.
    duration_s : float
        Clip duration in seconds.
    sample_rate : int
        Sampling rate in Hz.
    seed : int
        RNG seed; controls deterministic generation.
    noise_level : float
        SNR noise added to every sample (std of added Gaussian noise).
    """

    NUM_CLASSES = 4

    def __init__(
        self,
        n_samples: int = 2000,
        duration_s: float = 1.0,
        sample_rate: int = 16000,
        seed: int = 0,
        noise_level: float = 0.02,
    ):
        self.n_samples = n_samples
        self.duration_s = duration_s
        self.sample_rate = sample_rate
        self.seed = seed
        self.noise_level = noise_level
        self.length = int(duration_s * sample_rate)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Generate a (signal, label) pair deterministically from idx + seed."""
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        label = int(idx % self.NUM_CLASSES)  # balanced by construction
        x = self._generate(label, rng)
        # Add noise
        x = x + self.noise_level * rng.standard_normal(x.shape).astype(np.float32)
        # Normalize so peak is ~1.0
        x = x / (np.max(np.abs(x)) + 1e-8)
        return torch.from_numpy(x.astype(np.float32)), label

    def _generate(self, label: int, rng: np.random.Generator) -> np.ndarray:
        fs = self.sample_rate
        L = self.length
        t = np.arange(L) / fs

        if label == 0:
            # Pure chirp
            f0 = rng.uniform(100, 500)
            f1 = rng.uniform(2000, 5000)
            phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / self.duration_s * t**2)
            x = 0.5 * np.sin(phase)

        elif label == 1:
            # Chirp + click
            f0 = rng.uniform(100, 500)
            f1 = rng.uniform(2000, 5000)
            phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / self.duration_s * t**2)
            chirp = 0.5 * np.sin(phase)
            click_t = rng.uniform(0.2, 0.8)
            click = 1.5 * np.exp(-0.5 * ((t - click_t) / 0.001) ** 2)
            x = chirp + click

        elif label == 2:
            # Harmonic tone
            f0 = rng.uniform(80, 400)
            n_harmonics = rng.integers(3, 8)
            x = np.zeros_like(t)
            for k in range(1, n_harmonics + 1):
                if k * f0 >= fs / 2:
                    break
                x = x + (1.0 / k) * np.sin(2 * np.pi * k * f0 * t)

        elif label == 3:
            # White noise
            x = rng.standard_normal(L) * 0.3

        else:
            raise ValueError(f"Unknown label {label}")

        return x.astype(np.float32)