"""
ESC-50 environmental sound classification dataset loader.

ESC-50 = 2000 labeled audio clips of 5 seconds each at 44.1 kHz,
organized into 50 balanced classes (40 clips per class). A built-in
5-fold split is provided via the `fold` column in esc50.csv.

For our experiments:
  - We use fold=1 as test, folds 2-5 as train (1600/400 split).
  - Audio is resampled to 16 kHz to match the rest of the pipeline.
  - 5-second duration is kept fixed (ESC-50 is already uniform).

Label is the `target` integer column (0-49), with human-readable
names available in the `category` column.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm


@dataclass
class ESC50Sample:
    wav_path: Path
    label: int
    category: str
    fold: int


def _parse_csv(csv_path: Path) -> List[ESC50Sample]:
    """Parse esc50.csv into a list of ESC50Sample records."""
    samples = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        base_dir = csv_path.parent.parent / "audio"
        for row in reader:
            samples.append(ESC50Sample(
                wav_path=base_dir / row["filename"],
                label=int(row["target"]),
                category=row["category"],
                fold=int(row["fold"]),
            ))
    return samples


class ESC50Dataset(Dataset):
    """
    PyTorch Dataset for ESC-50 environmental sound classification.

    Parameters
    ----------
    root : str or Path
        Path to the ESC-50-master directory (containing audio/ and meta/).
    split : {'train', 'test'}
    test_fold : int in 1..5, default 1
        Which fold to use as the test set. The other 4 folds become train.
    target_sr : int, default 16000
        Resampling rate.
    duration_s : float, default 5.0
        Clip duration; ESC-50 is already 5 s so this is typically a no-op.
    cache_dir : str or Path or None, default None
        If provided, preprocessed 16 kHz waveforms are cached here.
    precompute : bool, default True
        If True and cache_dir is set, precomputes all waveforms at init.
    """

    NUM_CLASSES = 50

    def __init__(
        self,
        root,
        split: str,
        test_fold: int = 1,
        target_sr: int = 16000,
        duration_s: float = 5.0,
        cache_dir=None,
        precompute: bool = True,
    ):
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test'; got {split}")
        if test_fold not in {1, 2, 3, 4, 5}:
            raise ValueError(f"test_fold must be 1..5; got {test_fold}")

        self.root = Path(root)
        self.split = split
        self.test_fold = test_fold
        self.target_sr = target_sr
        self.duration_s = duration_s
        self.target_length = int(target_sr * duration_s)

        csv_path = self.root / "meta" / "esc50.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing ESC-50 metadata: {csv_path}")

        all_samples = _parse_csv(csv_path)
        if split == "test":
            self.samples = [s for s in all_samples if s.fold == test_fold]
        else:
            self.samples = [s for s in all_samples if s.fold != test_fold]

        # Build label -> category map for reporting
        self.category_names = {s.label: s.category for s in all_samples}

        if cache_dir is not None:
            self.cache_dir = Path(cache_dir) / f"sr{target_sr}_dur{duration_s}"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        if precompute and self.cache_dir is not None:
            self._precompute_all()

    def label_distribution(self) -> dict:
        counts = {}
        for s in self.samples:
            counts[s.label] = counts.get(s.label, 0) + 1
        return counts

    def _cache_path(self, idx: int) -> Path:
        s = self.samples[idx]
        key = s.wav_path.stem
        h = hashlib.sha1(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{h}.pt"

    def _precompute_all(self):
        to_build = [i for i in range(len(self.samples))
                    if not self._cache_path(i).exists()]
        if not to_build:
            return
        print(f"[ESC-50 {self.split}] caching {len(to_build)} clips "
              f"(test_fold={self.test_fold})...")
        for i in tqdm(to_build, desc=f"cache[{self.split}]"):
            self._load_raw(i)

    def _load_raw(self, idx: int) -> torch.Tensor:
        if self.cache_dir is not None:
            cp = self._cache_path(idx)
            if cp.exists():
                return torch.load(cp, weights_only=True)

        s = self.samples[idx]
        wav, sr = torchaudio.load(str(s.wav_path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)

        wav = wav.squeeze(0)

        # Pad or crop to fixed length (ESC-50 clips are 5s, should need no change
        # but we enforce it for robustness)
        L = wav.shape[0]
        if L < self.target_length:
            wav = torch.nn.functional.pad(wav, (0, self.target_length - L))
        elif L > self.target_length:
            offset = (L - self.target_length) // 2
            wav = wav[offset:offset + self.target_length]

        # Peak normalize
        peak = wav.abs().max()
        if peak > 1e-8:
            wav = wav / peak

        wav = wav.float()
        if self.cache_dir is not None:
            torch.save(wav, self._cache_path(idx))

        return wav

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x = self._load_raw(idx)
        y = self.samples[idx].label
        return x, y