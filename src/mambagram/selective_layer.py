"""
Selective MambaGram layer: input-dependent timescale Delta_t.

This is the "selectivity" that gives MambaGram its name. Unlike the
baseline MambaGramLayer (where the eigenvalues a_d are constant in
time), here we make the discretization timescale Delta_t a learned
function of the input x_t.

Mathematical model
------------------
Underlying continuous-time SSM (per channel d):
    dh_d/dt = (alpha_d + i*omega_d) h_d + b_d * x

Zero-order-hold discretization with timescale Delta_t:
    a_t^{(d)} = exp(Delta_t * (alpha_d + i*omega_d))
    h_t^{(d)} = a_t^{(d)} * h_{t-1}^{(d)} + Delta_t * b_d * x_t

where Delta_t = softplus(Linear(x_t)) is a positive scalar function
of the input. Larger Delta -> longer effective window per step;
smaller Delta -> shorter effective window.

Interpretation
--------------
On stationary regions, Delta_t grows large, the effective Gabor window
stretches, and frequency resolution sharpens (STFT-like behavior).
On transients, Delta_t shrinks, the window contracts, and time
resolution sharpens (scalogram-like behavior).

Implementation
--------------
Because A is now time-dependent (a_t depends on Delta_t which depends on
x_t), we cannot use FFT convolution. We use the explicit recurrent scan,
or — once we adapt to it — Mamba's parallel selective_scan_fn.

For this first version, we use the recurrent scan, which is fine for
short clips (< 2 s on a single GPU) and lets us validate the math.
A faster scan is added later.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveMambaGramLayer(nn.Module):
    """
    Selective (input-dependent) MambaGram layer.

    Parameters
    ----------
    n_channels : int
        Number of frequency channels D.
    sample_rate : int
        Audio sampling rate in Hz.
    f_min, f_max : float
        Frequency range for Mel-spaced initialization of omega_d.
    window_ms : float
        Initial effective window length in milliseconds (sets initial alpha_d).
    delta_min : float, default 0.1
        Lower bound for the data-dependent timescale Delta_t (in unit timesteps).
        Prevents Delta from collapsing to zero (which would give no integration).
    delta_max : float, default 10.0
        Upper bound for Delta_t. Prevents instability.
    delta_hidden : int, default 16
        Hidden width of the small MLP that maps x_t -> Delta_t.

    Inputs
    ------
    x : (batch, length) real float32 tensor.

    Returns
    -------
    h : (batch, length, n_channels) complex tensor.
    """

    def __init__(
        self,
        n_channels: int = 64,
        sample_rate: int = 16000,
        f_min: float = 80.0,
        f_max: Optional[float] = None,
        window_ms: float = 25.0,
        delta_min: float = 0.1,
        delta_max: float = 10.0,
        delta_hidden: int = 16,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate / 2.0
        self.window_ms = window_ms
        self.delta_min = delta_min
        self.delta_max = delta_max

        # --- Same Mel-spaced init as baseline layer ---
        freqs_hz = self._mel_spaced(n_channels, self.f_min, self.f_max)
        omega_init = 2.0 * math.pi * freqs_hz / sample_rate

        # window_samples controls *base* alpha; Delta_t modulates it
        window_samples = window_ms * 1e-3 * sample_rate
        c = 5.0
        alpha_init = torch.full((n_channels,), -c / window_samples)

        # Underlying continuous-time eigenvalues (learnable)
        self.omega = nn.Parameter(omega_init)
        # Parameterize alpha via softplus for stability (alpha < 0 always)
        raw_alpha_init = torch.log(torch.expm1(-alpha_init))
        self.raw_alpha = nn.Parameter(raw_alpha_init)

        # Input projection b_d (complex, parameterized as two real vectors)
        b_mag = 1.0 / math.sqrt(n_channels)
        self.b_real = nn.Parameter(torch.full((n_channels,), b_mag))
        self.b_imag = nn.Parameter(torch.zeros(n_channels))

        # --- Selectivity: small MLP mapping x_t to Delta_t ---
        # Input dim is 1 (scalar audio sample). Output dim is 1 (scalar Delta).
        # Could also be per-channel (output dim = D); we start with shared.
        self.delta_mlp = nn.Sequential(
            nn.Linear(1, delta_hidden),
            nn.SiLU(),
            nn.Linear(delta_hidden, 1),
        )
        # Initialize so that softplus(out) starts near 1.0
        # (i.e., Delta_t = 1.0 initially; pure baseline behavior)
        nn.init.zeros_(self.delta_mlp[-1].weight)
        nn.init.constant_(self.delta_mlp[-1].bias, math.log(math.expm1(1.0)))

    @staticmethod
    def _mel_spaced(n: int, f_min: float, f_max: float) -> torch.Tensor:
        def hz_to_mel(f): return 2595.0 * math.log10(1.0 + f / 700.0)
        def mel_to_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        m_min, m_max = hz_to_mel(f_min), hz_to_mel(f_max)
        mels = torch.linspace(m_min, m_max, n)
        return torch.tensor([mel_to_hz(m.item()) for m in mels])

    def _get_alpha(self) -> torch.Tensor:
        """Negative alpha for stability."""
        return -F.softplus(self.raw_alpha)

    def _compute_delta(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute input-dependent timescale Delta_t.

        Parameters
        ----------
        x : (batch, length) real tensor.

        Returns
        -------
        delta : (batch, length) positive real tensor in [delta_min, delta_max].
        """
        # MLP expects shape (batch, length, 1)
        x_in = x.unsqueeze(-1)                                # (B, L, 1)
        d_raw = self.delta_mlp(x_in).squeeze(-1)              # (B, L)
        # Bound Delta_t softly to [delta_min, delta_max]
        delta = self.delta_min + (self.delta_max - self.delta_min) * torch.sigmoid(d_raw)
        return delta

    def forward(self, x: torch.Tensor, return_delta: bool = False):
        """
        Run the selective scan.

        Parameters
        ----------
        x : (batch, length) real float32 tensor.
        return_delta : bool
            If True, also return the Delta_t trajectory (useful for visualization).

        Returns
        -------
        h : (batch, length, n_channels) complex tensor.
        delta : (batch, length) tensor (only if return_delta=True).
        """
        if x.dim() != 2:
            raise ValueError(f"Expected (batch, length); got {x.shape}")

        batch, length = x.shape
        D = self.n_channels
        device = x.device
        dtype = x.dtype

        # Continuous-time eigenvalues (D,)
        alpha = self._get_alpha().to(device=device, dtype=dtype)      # (D,)
        omega = self.omega.to(device=device, dtype=dtype)             # (D,)
        # log_a is the per-channel continuous-time eigenvalue
        log_a = torch.complex(alpha, omega)                           # (D,)

        # Input projection b_d (D,)
        b = torch.complex(self.b_real, self.b_imag).to(device=device) # (D,)

        # Per-step input-dependent timescale (B, L)
        delta = self._compute_delta(x)                                # (B, L)

        # Initialize hidden state
        h_prev = torch.zeros(batch, D, dtype=torch.complex64, device=device)
        H = torch.zeros(batch, length, D, dtype=torch.complex64, device=device)
        x_c = x.to(torch.complex64)                                   # (B, L)

        # Selective recurrent scan
        for t in range(length):
            # Discretize: a_t^(d) = exp(delta_t * log_a^(d))
            # delta[:, t] has shape (B,); broadcast to (B, D)
            d_t = delta[:, t].unsqueeze(-1).to(torch.complex64)       # (B, 1)
            a_t = torch.exp(d_t * log_a[None, :])                     # (B, D)
            # Discretized B: delta * b
            b_t = d_t * b[None, :]                                    # (B, D)

            h_prev = a_t * h_prev + b_t * x_c[:, t:t+1]
            H[:, t, :] = h_prev

        if return_delta:
            return H, delta
        return H

    def magnitude(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).abs()
