"""Suppress known harmless warnings from mamba-ssm + PyTorch version mismatch."""
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.custom_(fwd|bwd).*is deprecated.*",
    category=FutureWarning,
)
