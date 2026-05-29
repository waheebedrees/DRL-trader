from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# NoisyLinear — learned exploration via factorised Gaussian noise
# ─────────────────────────────────────────────────────────────────────────────


class NoisyLinear(nn.Module):
    """
    NoisyNet linear layer.
    Replaces epsilon-greedy — noise magnitude is learned by gradient descent.
    During eval() noise is zeroed for deterministic inference.

    Reference: Fortunato et al. (2017) "Noisy Networks for Exploration"
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        std_init:     float = 0.5,
        factorised:   bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.factorised = factorised

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer(
            "weight_eps", torch.zeros(out_features, in_features))
        self.register_buffer("bias_eps",   torch.zeros(out_features))

        self._reset_parameters()
        self.reset_noise()

    def _reset_parameters(self) -> None:
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _f(x: Tensor) -> Tensor:
        """Factorised noise transform: sign(x) * sqrt(|x|)."""
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        """Resample noise buffers — call once per update step."""
        if self.factorised:
            eps_i = self._f(torch.randn(self.in_features,
                            device=self.weight_mu.device))
            eps_j = self._f(torch.randn(self.out_features,
                            device=self.weight_mu.device))
            self.weight_eps.copy_(eps_j.outer(eps_i))
            self.bias_eps.copy_(eps_j)
        else:
            self.weight_eps.normal_()
            self.bias_eps.normal_()

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"std_init={self.std_init}, factorised={self.factorised}")


# ─────────────────────────────────────────────────────────────────────────────
# Gated Residual Network block
# ─────────────────────────────────────────────────────────────────────────────

class GatedResidualBlock(nn.Module):
    """
    GRN from Temporal Fusion Transformer.
    Acts as skip connection with learned gating — the gate
    learns when residual path is useful vs identity.
    """

    def __init__(self, din: int, dout: int, drop: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(din)
        self.fc1 = nn.Linear(din, din * 2)
        self.fc2 = nn.Linear(din * 2, dout)
        self.gate = nn.Linear(din, dout)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()
        self.proj = nn.Linear(
            din, dout, bias=False) if din != dout else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # Auto-handle 2D vector input [B, D]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # → [B, 1, D]
            was_2d = True
        else:
            was_2d = False

        h = self.act(self.fc1(self.norm(x)))
        h = self.fc2(self.drop(h))
        g = torch.sigmoid(self.gate(x.squeeze(1) if was_2d else x))
        if was_2d:
            g = g.unsqueeze(1)
        out = g * h + (1 - g) * self.proj(x)

        return out.squeeze(1) if was_2d else out
# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal + Learnable Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────


class PositionalEncoding(nn.Module):
    """
    Hybrid sinusoidal + learned positional encoding.
    Sinusoidal provides structure; learned part adapts to market patterns.
    """

    def __init__(
        self,
        d_model:  int,
        max_len:  int = 512,
        dropout:  float = 0.1,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.learnable = learnable

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe_fixed", pe.unsqueeze(0))  # [1, max_len, D]

        if learnable:
            self.pe_learned = nn.Parameter(
                torch.zeros(1, max_len, d_model) * 0.01)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, D]
        T = x.size(1)
        enc = self.pe_fixed[:, :T, :]
        if self.learnable:
            enc = enc + self.pe_learned[:, :T, :]
        return self.dropout(x + enc)


# ─────────────────────────────────────────────────────────────────────────────
# Rotary Positional Embedding (RoPE)
# ─────────────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    RoPE — encodes relative position directly into attention scores.
    Better generalisation to unseen sequence lengths.
    Reference: Su et al. (2021) "RoFormer"
    """

    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        inv_freq = 1.0 / (10_000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0))

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        # q, k: [B, heads, T, head_dim]
        T = q.size(2)

        # 🔧 FIX: Rebuild cache if sequence exceeds current cache
        if T > self.cos_cached.size(2):
            self._build_cache(T)

        cos = self.cos_cached[:, :, :T, :]
        sin = self.sin_cached[:, :, :T, :]
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k


# ─────────────────────────────────────────────────────────────────────────────
# Market Attention — causal multi-head attention for time-series
# ─────────────────────────────────────────────────────────────────────────────

class MarketAttention(nn.Module):
    """
    Causal multi-head attention specialised for time-series.

    Features:
    - RoPE positional embedding (relative, not absolute)
    - Optional causal mask (prevents future leakage)
    - Per-head scaling
    - Flash-attention compatible layout
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        dropout:  float = 0.1,
        causal:   bool = True,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim)
        else:
            self.rope = None

    def forward(
        self,
        x:    Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, T, D = x.shape

        # Pre-norm
        h = self.norm(x)

        Q = self.q_proj(h).view(B, T, self.n_heads,
                                self.head_dim).transpose(1, 2)
        K = self.k_proj(h).view(B, T, self.n_heads,
                                self.head_dim).transpose(1, 2)
        V = self.v_proj(h).view(B, T, self.n_heads,
                                self.head_dim).transpose(1, 2)
        # Q, K, V: [B, heads, T, head_dim]

        if self.rope is not None:
            Q, K = self.rope(Q, K)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Causal mask
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)

        # 🔧 FIX: Handle NaN in attention weights (safety for extreme negative scores)
        weights = torch.nan_to_num(weights, nan=0.0)

        weights = self.dropout(weights)

        # [B, heads, T, head_dim]
        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        return x + out   # residual connection


# ─────────────────────────────────────────────────────────────────────────────
# Feed-Forward with SwiGLU activation
# ─────────────────────────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward (used in LLaMA, PaLM).
    Gate mechanism provides stronger non-linearity than standard FFN.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.proj = nn.Linear(d_model, d_ff, bias=False)
        self.out_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        gate_val = F.silu(self.gate(h))
        proj_val = self.proj(h)
        h = self.dropout(gate_val * proj_val)
        h = self.out_proj(h)
        return x + h   # residual


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Convolutional Block (dilated causal convolutions)
# ─────────────────────────────────────────────────────────────────────────────

class DilatedCausalConv(nn.Module):
    """
    Dilated causal 1D convolution block.
    Efficient receptive field growth: 2^layer dilation.
    Captures both short-term micro-structure and longer patterns.
    """

    def __init__(
        self,
        channels:  int,
        kernel:    int = 3,
        dilation:  int = 1,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()
        self.padding = (kernel - 1) * dilation

        self.norm = nn.LayerNorm(channels)
        self.conv1 = nn.Conv1d(
            channels, channels * 2,
            kernel_size=kernel, dilation=dilation,
            padding=self.padding,
        )
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, C] — convert to [B, C, T] for conv
        residual = x
        h = self.norm(x).transpose(1, 2)     # [B, C, T]

        # Dilated conv + remove future frames (causal)
        h = self.conv1(h)
        if self.padding > 0:
            h = h[:, :, :-self.padding]       # remove future

        # GLU gating
        gate, val = h.chunk(2, dim=1)
        h = torch.sigmoid(gate) * val         # [B, C, T]
        h = self.drop(h)
        h = self.conv2(h)
        h = h.transpose(1, 2)                 # [B, T, C]
        return h + residual


# ─────────────────────────────────────────────────────────────────────────────
# Squeeze-and-Excitation channel attention
# ─────────────────────────────────────────────────────────────────────────────

class SqueezeExcitation(nn.Module):
    """
    SE block for feature recalibration.
    Learns which features are most informative at each step.
    
    🔧 FIX: Uses standard SE formulation: output = x * scale
    Original paper: Hu et al. (2018) "Squeeze-and-Excitation Networks"
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, C]
        scale = self.se(x.mean(dim=1, keepdim=True))  # [B, 1, C]
        return x * scale



