"""MambaGram: Adaptive Time-Frequency Representations via Selective State Space Models."""

from mambagram._warnings import *
from mambagram.layers import MambaGramLayer
from mambagram.selective_layer import SelectiveMambaGramLayer
from mambagram.gated_layer import GatedMambaGramLayer
from mambagram.datasets import SyntheticAudioDataset

__version__ = "0.0.4"
__all__ = [
    "MambaGramLayer",
    "SelectiveMambaGramLayer",
    "GatedMambaGramLayer",
    "SyntheticAudioDataset",
]

from mambagram.classifier import AudioClassifier

__all__ += ["AudioClassifier"]