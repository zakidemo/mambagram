"""
Gated MambaGram layer: input-dependent forgetting with fixed Gabor grid.

This is the CORRECT selectivity mechanism for MambaGram, replacing the
failed Δ-based approach. Motivation: Δ_t rescales the underlying
frequency grid and destroys the filter-bank interpretation. A forgetting
gate γ_t ∈ [0, 1] preserves the grid while still enabling input-
dependent adaptation.

Mathematical model
------------------
Per channel d and time t:
    γ_t^{(d)} = sigmoid(MLP(x_t))                    ∈ [0, 1]
    h_t^{(d)} = γ_t^{(d)} * a_d * h_{t-1}^{(d)} + b_d * x_t

where a_d = exp(α_d + i·ω_d) is the FIXED per-channel eigenvalue
(Gabor-grid preserved), and γ is shared across channels or per-channel.

Interpretation
--------------
  γ = 1 → baseline behavior (full memory, nominal window)
  γ = 0 → state reset (zero memory, single-sample response)
  γ ∈ (0, 1) → partial forgetting (shorter effective window)

Physically, γ = γ_t is a "reset valve" that can briefly flush hidden
state when a transient arrives. After the reset, the channel resumes
accumulating, producing a sharp temporal localization of the event.

Critically, the center frequency ω_d and nominal decay α_d are unchanged.
Only the *effective memory depth* is modulated.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMambaGramLayer(nn.Module):
    """
    MambaGram with input-dependent forgetting gate γ_t.

    Parameters
    ----------
    n_channels : int
        Number of frequency channels D.
    sample_rate : int
    f_min, f_max : float
        Frequency range for Mel-spaced init.
    window_ms : float
        Nominal (maximum) window length in milliseconds.
    gate_hidden : int, default 16
        Hidden width of the MLP that maps x_t to γ_t.
    gate_per_channel : bool, default False
        If True, γ has shape (D,) — each channel gets its own gate.
        If False, γ is scalar per time step, shared across all channels.
    gate_smooth_ms : float, default 1.0
        Apply a moving-average filter of this width (ms) to the gate
        before applying. Ensures smooth variation in time. Set to 0 to disable.

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
        gate_hidden: int = 16,
        gate_per_channel: bool = False,
        gate_smooth_ms: float = 1.0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate / 2.0
        self.window_ms = window_ms
        self.gate_per_channel = gate_per_channel
        self.gate_smooth_ms = gate_smooth_ms

        # --- Mel-spaced frequency init (fixed grid; preserved regardless of γ) ---
        freqs_hz = self._mel_spaced(n_channels, self.f_min, self.f_max)
        omega_init = 2.0 * math.pi * freqs_hz / sample_rate

        window_samples = window_ms * 1e-3 * sample_rate
        c = 5.0
        alpha_init = torch.full((n_channels,), -c / window_samples)

        self.omega = nn.Parameter(omega_init)
        # Parameterize α via softplus for stability (α < 0 guaranteed)
        raw_alpha_init = torch.log(torch.expm1(-alpha_init))
        self.raw_alpha = nn.Parameter(raw_alpha_init)

        # Input projection b_d (complex)
        b_mag = 1.0 / math.sqrt(n_channels)
        self.b_real = nn.Parameter(torch.full((n_channels,), b_mag))
        self.b_imag = nn.Parameter(torch.zeros(n_channels))

        # --- Gate MLP: x_t -> γ_t ---
        gate_out_dim = n_channels if gate_per_channel else 1
        self.gate_mlp = nn.Sequential(
            nn.Linear(1, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, gate_out_dim),
        )
        # Initialize so sigmoid(bias) ≈ 1.0 → γ starts near 1.0 (baseline behavior)
        # We use bias = 5.0, so sigmoid(5.0) ≈ 0.993
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, 5.0)

    @staticmethod
    def _mel_spaced(n: int, f_min: float, f_max: float) -> torch.Tensor:
        def hz_to_mel(f): return 2595.0 * math.log10(1.0 + f / 700.0)
        def mel_to_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        m_min, m_max = hz_to_mel(f_min), hz_to_mel(f_max)
        mels = torch.linspace(m_min, m_max, n)
        return torch.tensor([mel_to_hz(m.item()) for m in mels])

    def _get_alpha(self) -> torch.Tensor:
        return -F.softplus(self.raw_alpha)

    def _compute_gate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute γ_t from x_t.

        Parameters
        ----------
        x : (batch, length) real tensor.

        Returns
        -------
        gamma : (batch, length) if gate_per_channel=False,
                (batch, length, n_channels) if gate_per_channel=True.
                Values in [0, 1].
        """
        x_in = x.unsqueeze(-1)                                   # (B, L, 1)
        raw = self.gate_mlp(x_in)                                # (B, L, out_dim)
        gamma = torch.sigmoid(raw)                               # (B, L, out_dim)

        # Smooth the gate in time to avoid rapid oscillation
        if self.gate_smooth_ms > 0:
            win = max(1, int(self.gate_smooth_ms * 1e-3 * self.sample_rate))
            if win > 1:
                # conv1d expects (B, C, L); we have (B, L, C)
                g = gamma.transpose(1, 2)                        # (B, C, L)
                pad = win // 2
                kernel = torch.ones(g.size(1), 1, win, device=g.device) / win
                # Per-channel smoothing via grouped conv
                g_padded = F.pad(g, (pad, pad), mode="reflect")
                g_smooth = F.conv1d(g_padded, kernel, groups=g.size(1))
                # Trim to original length (pad may over-produce by 1 if win is even)
                g_smooth = g_smooth[..., :gamma.size(1)]
                gamma = g_smooth.transpose(1, 2)                 # (B, L, C)

        if not self.gate_per_channel:
            gamma = gamma.squeeze(-1)                            # (B, L)

        return gamma

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        """
        Run the gated recurrent scan.

        Parameters
        ----------
        x : (batch, length) real float32 tensor.
        return_gate : bool
            If True, also return γ_t for inspection.

        Returns
        -------
        h : (batch, length, n_channels) complex tensor.
        gate : (batch, length) or (batch, length, n_channels) — only if return_gate.
        """
        if x.dim() != 2:
            raise ValueError(f"Expected (batch, length); got {x.shape}")

        batch, length = x.shape
        D = self.n_channels
        device = x.device
        dtype = x.dtype

        # Fixed Gabor-grid eigenvalues
        alpha = self._get_alpha().to(device=device, dtype=dtype)
        omega = self.omega.to(device=device, dtype=dtype)
        a = torch.complex(alpha, omega).exp()                    # (D,) complex
        b = torch.complex(self.b_real, self.b_imag).to(device=device)  # (D,) complex

        # Input-dependent gate
        gamma = self._compute_gate(x)                            # (B, L) or (B, L, D)

        # Prepare gamma for broadcast: we always want (B, L, D)
        if gamma.dim() == 2:
            gamma = gamma.unsqueeze(-1).expand(-1, -1, D)        # (B, L, D)

        # Recurrent scan
        h_prev = torch.zeros(batch, D, dtype=torch.complex64, device=device)
        H = torch.zeros(batch, length, D, dtype=torch.complex64, device=device)
        x_c = x.to(torch.complex64)                              # (B, L)

        for t in range(length):
            gamma_t = gamma[:, t, :].to(torch.complex64)         # (B, D)
            h_prev = gamma_t * a[None, :] * h_prev + b[None, :] * x_c[:, t:t+1]
            H[:, t, :] = h_prev

        if return_gate:
            gamma_out = gamma if self.gate_per_channel else gamma[..., 0]
            return H, gamma_out
        return H

    def magnitude(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).abs()
