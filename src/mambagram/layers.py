"""
MambaGram layers: learnable, complex-valued time-frequency representations
built on diagonal state space models.

This module implements the core MambaGramLayer: a differentiable, complex-valued
filter bank where each channel is parameterized as a discrete-time Gabor-like
atom. Under mild conditions on the parameters (alpha < 0), the hidden state
trajectory of each channel coincides with the analytic signal of the input
in a frequency band centered at omega_d.

Mathematical model
------------------
For each channel d and time step t:

    h_t^{(d)} = a_d * h_{t-1}^{(d)} + b_d * x_t

where a_d = exp(alpha_d + i * omega_d) is the learnable complex eigenvalue.

The unrolled response is a convolution of x with a complex exponentially-
decaying kernel g_d(tau) = b_d * exp((alpha_d + i*omega_d) * tau), which is a
Gabor atom. See the project paper for full derivation.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class MambaGramLayer(nn.Module):
    """
    Diagonal complex-valued state space layer producing a learnable spectrogram.

    Parameters
    ----------
    n_channels : int
        Number of frequency channels D (analogous to number of Mel bins).
    sample_rate : int
        Audio sampling rate in Hz. Used only to map center frequencies to
        normalized angular frequencies omega_d = 2*pi*f_d / fs.
    f_min : float, default 40.0
        Minimum center frequency in Hz for initialization.
    f_max : float or None, default None
        Maximum center frequency in Hz. Defaults to sample_rate / 2 (Nyquist).
    window_ms : float, default 25.0
        Target effective window length in milliseconds for the decay envelope.
    init : {'mel', 'linear'}, default 'mel'
        Frequency spacing for initialization of omega_d.
    trainable_freq : bool, default True
        If True, omega_d is a learnable parameter; otherwise fixed.
    trainable_decay : bool, default True
        If True, alpha_d is learnable; otherwise fixed.

    Inputs
    ------
    x : torch.Tensor of shape (batch, length)
        Real-valued waveform, float32.

    Returns
    -------
    h : torch.Tensor of shape (batch, length, n_channels)
        Complex-valued hidden states. Use `h.abs()` as the "MambaGram"
        magnitude spectrogram.
    """

    def __init__(
        self,
        n_channels: int = 64,
        sample_rate: int = 16000,
        f_min: float = 40.0,
        f_max: Optional[float] = None,
        window_ms: float = 25.0,
        init: str = "mel",
        trainable_freq: bool = True,
        trainable_decay: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate / 2.0
        self.window_ms = window_ms

        # --- Initialize center frequencies ---
        if init == "mel":
            freqs_hz = self._mel_spaced_frequencies(n_channels, self.f_min, self.f_max)
        elif init == "linear":
            freqs_hz = torch.linspace(self.f_min, self.f_max, n_channels)
        else:
            raise ValueError(f"Unknown init='{init}'. Use 'mel' or 'linear'.")

        # Convert to angular frequencies (radians per sample)
        omega_init = 2.0 * math.pi * freqs_hz / sample_rate  # shape (D,)

        # --- Initialize decay rate ---
        # We want |a_d|^(window_samples) ~ exp(-c), i.e. alpha * window_samples = -c
        # Picking c = 5 gives a reasonable Hann-like envelope decay.
        window_samples = window_ms * 1e-3 * sample_rate
        c = 5.0
        alpha_init = torch.full((n_channels,), -c / window_samples)

        # --- Register as parameters or buffers ---
        if trainable_freq:
            self.omega = nn.Parameter(omega_init)
        else:
            self.register_buffer("omega", omega_init)

        if trainable_decay:
            # Parameterize alpha as -softplus(raw_alpha) to guarantee alpha < 0
            # Inverse softplus of |alpha_init| for initialization
            raw_alpha_init = torch.log(torch.expm1(-alpha_init))  # -alpha_init > 0
            self.raw_alpha = nn.Parameter(raw_alpha_init)
        else:
            self.register_buffer("alpha", alpha_init)

        self._trainable_decay = trainable_decay

        # --- Input projection b_d, initialized for unit gain ---
        # b is complex. We parameterize as two real vectors for PyTorch compatibility.
        b_mag = 1.0 / math.sqrt(n_channels)
        b_phase = torch.zeros(n_channels)
        self.b_real = nn.Parameter(torch.full((n_channels,), b_mag) * torch.cos(b_phase))
        self.b_imag = nn.Parameter(torch.full((n_channels,), b_mag) * torch.sin(b_phase))

    @staticmethod
    def _mel_spaced_frequencies(n: int, f_min: float, f_max: float) -> torch.Tensor:
        """Return n frequencies in Hz spaced on a Mel scale between f_min and f_max."""
        def hz_to_mel(f):
            return 2595.0 * math.log10(1.0 + f / 700.0)

        def mel_to_hz(m):
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        m_min, m_max = hz_to_mel(f_min), hz_to_mel(f_max)
        mels = torch.linspace(m_min, m_max, n)
        return torch.tensor([mel_to_hz(m.item()) for m in mels])

    def _get_alpha(self) -> torch.Tensor:
        """Return alpha ensuring alpha < 0 (stability)."""
        if self._trainable_decay:
            return -torch.nn.functional.softplus(self.raw_alpha)
        return self.alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the diagonal complex SSM over a batch of waveforms.

        Parameters
        ----------
        x : (batch, length) real float32 tensor.

        Returns
        -------
        h : (batch, length, n_channels) complex tensor.
        """
        if x.dim() != 2:
            raise ValueError(f"Expected input of shape (batch, length); got {x.shape}")

        batch, length = x.shape
        D = self.n_channels
        device = x.device
        dtype = x.dtype

        # Build complex eigenvalues a_d = exp(alpha_d + i * omega_d)
        alpha = self._get_alpha().to(device=device, dtype=dtype)      # (D,)
        omega = self.omega.to(device=device, dtype=dtype)             # (D,)
        # a_d in complex form: (D,)
        a = torch.complex(alpha, omega).exp()                         # (D,) complex

        # Input projection b_d: (D,) complex
        b = torch.complex(self.b_real, self.b_imag).to(device=device) # (D,) complex

        # Initialize hidden state
        h_prev = torch.zeros(batch, D, dtype=torch.complex64, device=device)

        # Output buffer
        H = torch.zeros(batch, length, D, dtype=torch.complex64, device=device)

        # Cast x to complex for the recurrence
        x_c = x.to(torch.complex64)  # (batch, length)

        # Recurrent scan (O(L*D) time, O(D) memory per step)
        # Broadcasting: a is (D,), b is (D,), x_c[:, t] is (batch,)
        for t in range(length):
            # h_t^{(d)} = a_d * h_{t-1}^{(d)} + b_d * x_t
            h_prev = a[None, :] * h_prev + b[None, :] * x_c[:, t:t + 1]
            H[:, t, :] = h_prev

        return H

    def magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Return |h| as a real-valued spectrogram of shape (batch, length, n_channels)."""
        return self.forward(x).abs()
