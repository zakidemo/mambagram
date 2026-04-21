"""
ICBHI 2017 Respiratory Sound Dataset loader.

Loads the ICBHI 2017 database from a local directory containing:
  - ~920 .wav files  (audio recordings, heterogeneous sample rates)
  - ~920 .txt files  (4-column breath-cycle annotations)
  - ICBHI_challenge_train_test.txt  (official patient-level split)

Each breath cycle (as annotated) becomes one labeled sample. Labels
follow the standard 4-class ICBHI protocol:
  0 = Normal    (crackle=0, wheeze=0)
  1 = Crackle   (crackle=1, wheeze=0)
  2 = Wheeze    (crackle=0, wheeze=1)
  3 = Both      (crackle=1, wheeze=1)

Audio is resampled to a common sample rate (default 16 kHz) and padded
or center-cropped to a fixed duration (default 8 s).

A disk cache of preprocessed float32 waveforms is built on first use to
speed up subsequent epochs.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm


# --- Label convention ---
LABEL_NORMAL = 0
LABEL_CRACKLE = 1
LABEL_WHEEZE = 2
LABEL_BOTH = 3
LABEL_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]


@dataclass
class BreathCycle:
    """One annotated breath cycle = one dataset sample."""
    wav_path: Path
    start_s: float
    end_s: float
    label: int          # 0/1/2/3
    patient_id: int     # e.g. 101
    file_stem: str      # e.g. "101_1b1_Al_sc_Meditron"
    split: str          # "train" or "test"


def _parse_annotation_file(txt_path: Path) -> List[Tuple[float, float, int, int]]:
    """
    Parse an ICBHI annotation file.

    Each line is: start_s  end_s  crackle  wheeze   (whitespace separated)
    Returns list of (start, end, crackle, wheeze) tuples.
    """
    cycles = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                s, e, c, w = float(parts[0]), float(parts[1]), int(parts[2]), int(parts[3])
                cycles.append((s, e, c, w))
            except ValueError:
                continue
    return cycles


def _cycle_label(crackle: int, wheeze: int) -> int:
    """4-class label from binary crackle/wheeze flags."""
    if crackle == 0 and wheeze == 0:
        return LABEL_NORMAL
    if crackle == 1 and wheeze == 0:
        return LABEL_CRACKLE
    if crackle == 0 and wheeze == 1:
        return LABEL_WHEEZE
    return LABEL_BOTH


def _parse_split_file(split_path: Path) -> dict:
    """
    Parse ICBHI_challenge_train_test.txt.

    Each line: <file_stem>\t<train|test>
    Returns dict mapping file_stem -> 'train' or 'test'.
    """
    mapping = {}
    with open(split_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            mapping[parts[0]] = parts[1]
    return mapping


def scan_icbhi_cycles(root: Path) -> List[BreathCycle]:
    """
    Walk the ICBHI directory and collect all breath cycles with their labels
    and train/test split assignment.

    Parameters
    ----------
    root : Path
        Directory containing the .wav, .txt, and ICBHI_challenge_train_test.txt files.

    Returns
    -------
    List of BreathCycle records.
    """
    root = Path(root)
    split_file = root / "ICBHI_challenge_train_test.txt"
    if not split_file.exists():
        raise FileNotFoundError(
            f"Missing official split file: {split_file}\n"
            "Download it from:\n"
            "  https://bhichallenge.med.auth.gr/sites/default/files/"
            "ICBHI_final_database/ICBHI_challenge_train_test.txt"
        )
    split_map = _parse_split_file(split_file)

    cycles = []
    wav_files = sorted(root.glob("*.wav"))
    skipped = 0
    for wav_path in wav_files:
        stem = wav_path.stem
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            skipped += 1
            continue
        split = split_map.get(stem)
        if split is None:
            skipped += 1
            continue

        # Patient ID is the first underscore-separated token
        m = re.match(r"^(\d+)_", stem)
        patient_id = int(m.group(1)) if m else -1

        for (s, e, c, w) in _parse_annotation_file(txt_path):
            if e <= s:
                continue
            cycles.append(BreathCycle(
                wav_path=wav_path,
                start_s=s, end_s=e,
                label=_cycle_label(c, w),
                patient_id=patient_id,
                file_stem=stem,
                split=split,
            ))

    if skipped > 0:
        print(f"[ICBHI] skipped {skipped} files (missing txt or split entry)")
    return cycles


class ICBHIDataset(Dataset):
    """
    PyTorch Dataset for ICBHI 2017 breath-cycle classification.

    Parameters
    ----------
    root : str or Path
        Directory containing the ICBHI wav/txt files and split file.
    split : {'train', 'test'}
    target_sr : int, default 16000
        All audio is resampled to this rate.
    duration_s : float, default 8.0
        Fixed clip length in seconds; short cycles are zero-padded, long
        ones are center-cropped.
    cache_dir : str or Path or None, default None
        If provided, preprocessed waveforms are cached here as .pt files.
        Strongly recommended — first epoch builds the cache, later epochs
        load from it in milliseconds.
    precompute : bool, default True
        If True and cache_dir is set, precomputes all waveforms at
        construction time (with a tqdm progress bar). Otherwise, samples
        are preprocessed lazily on first access.
    """

    def __init__(
        self,
        root,
        split: str,
        target_sr: int = 16000,
        duration_s: float = 8.0,
        cache_dir=None,
        precompute: bool = True,
    ):
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test'; got {split}")

        self.root = Path(root)
        self.split = split
        self.target_sr = target_sr
        self.duration_s = duration_s
        self.target_length = int(target_sr * duration_s)

        # Build or load list of breath cycles
        all_cycles = scan_icbhi_cycles(self.root)
        self.cycles = [c for c in all_cycles if c.split == split]

        # Cache setup
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir) / f"sr{target_sr}_dur{duration_s}"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        if precompute and self.cache_dir is not None:
            self._precompute_all()

    # -------- Label distribution helper --------
    def label_distribution(self) -> dict:
        counts = {k: 0 for k in range(4)}
        for c in self.cycles:
            counts[c.label] += 1
        return counts

    # -------- Cache machinery --------
    def _cache_path(self, idx: int) -> Path:
        c = self.cycles[idx]
        key = f"{c.file_stem}_{c.start_s:.3f}_{c.end_s:.3f}"
        # Hash to keep filenames short
        h = hashlib.sha1(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{h}.pt"

    def _precompute_all(self):
        """Run the full dataset through _load_raw once, building the cache."""
        to_build = []
        for i in range(len(self.cycles)):
            if not self._cache_path(i).exists():
                to_build.append(i)
        if not to_build:
            return
        print(f"[ICBHI {self.split}] caching {len(to_build)} breath cycles "
              f"(one-time preprocessing)...")
        for i in tqdm(to_build, desc=f"cache[{self.split}]"):
            self._load_raw(i)  # side effect: writes to cache

    # -------- Core loading --------
    def _load_raw(self, idx: int) -> torch.Tensor:
        """
        Load, resample, and pad/crop one breath cycle.
        Caches the result to disk if cache_dir is set.
        """
        if self.cache_dir is not None:
            cache_path = self._cache_path(idx)
            if cache_path.exists():
                return torch.load(cache_path, weights_only=True)

        c = self.cycles[idx]

        # Load full wav (torchaudio returns float32 by default)
        wav, sr = torchaudio.load(str(c.wav_path))  # (channels, time)
        # Downmix to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Extract segment [start_s, end_s] at the ORIGINAL sample rate
        s_idx = int(round(c.start_s * sr))
        e_idx = int(round(c.end_s * sr))
        s_idx = max(0, s_idx)
        e_idx = min(wav.shape[1], e_idx)
        segment = wav[:, s_idx:e_idx]

        # Resample to target_sr
        if sr != self.target_sr:
            segment = torchaudio.functional.resample(segment, sr, self.target_sr)

        # Squeeze channel dim to 1D
        segment = segment.squeeze(0)  # (time,)

        # Pad or center-crop to fixed length
        L = segment.shape[0]
        if L < self.target_length:
            # Zero-pad on the right
            segment = torch.nn.functional.pad(segment, (0, self.target_length - L))
        elif L > self.target_length:
            # Center crop
            offset = (L - self.target_length) // 2
            segment = segment[offset:offset + self.target_length]

        # Peak normalize
        peak = segment.abs().max()
        if peak > 1e-8:
            segment = segment / peak

        # Cache
        if self.cache_dir is not None:
            torch.save(segment.float(), self._cache_path(idx))

        return segment.float()

    def __len__(self) -> int:
        return len(self.cycles)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x = self._load_raw(idx)
        y = self.cycles[idx].label
        return x, y