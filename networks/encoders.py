# networks/encoders.py

from __future__ import annotations
import math
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

from config.config import NetworkConfig
from state import Architecture

from networks.layers import (
    NoisyLinear,
    PositionalEncoding,
    RotaryEmbedding,
    MarketAttention,
    GatedResidualBlock,
    SqueezeExcitation,
    DilatedCausalConv,
    SwiGLUFFN,
)
# ─────────────────────────────────────────────────────────────────────────────
# Output containers
# ─────────────────────────────────────────────────────────────────────────────

class ActorOutput(NamedTuple):
    action:   Tensor          # sampled action
    log_prob: Tensor          # log probability of action
    entropy:  Tensor          # distribution entropy
    mean:     Tensor          # distribution mean (for analysis)
    std:      Tensor          # distribution std


class CriticOutput(NamedTuple):
    value:    Tensor          # state value V(s) or Q(s,a)


class DistributionalOutput(NamedTuple):
    log_probs: Tensor         # [B, A, n_atoms] log distribution
    q_values:  Tensor         # [B, A] expected Q


# ─────────────────────────────────────────────────────────────────────────────
# AttentionLSTM — LSTM with multi-head temporal attention
# ─────────────────────────────────────────────────────────────────────────────


class MarketTransformerEncoder(nn.Module):
    """
    Full transformer encoder over market bar sequences.

    Architecture:
        Linear projection → PositionalEncoding
        → N × (CausalSelfAttention + SwiGLUFFN + GRN)
        → CLS-token aggregation → LayerNorm output

    CLS token aggregates full sequence into fixed-size representation.
    RoPE handles relative temporal positions natively.
    """

    def __init__(self, cfg: NetworkConfig, n_features: int) -> None:
        super().__init__()
        D = cfg.d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, D),
            nn.LayerNorm(D),
            nn.Dropout(cfg.dropout),
        )

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, D) * 0.02)

        # Positional encoding
        self.pos_enc = PositionalEncoding(
            d_model=D,
            max_len=cfg.max_seq_len + 1,
            dropout=cfg.dropout,
            learnable=True,
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": MarketAttention(
                    d_model=D,
                    n_heads=cfg.n_heads,
                    dropout=cfg.dropout,
                    causal=True,
                    use_rope=True,
                ),
                "ffn":  SwiGLUFFN(D, cfg.d_ff, cfg.dropout),
                "grn":  GatedResidualBlock(D, D, drop=cfg.dropout),

                "se":   SqueezeExcitation(D),
            })
            for _ in range(cfg.n_layers)
        ])

        self.norm = nn.LayerNorm(D)
        self.out_dim = D

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, F]
        B = x.size(0)

        h = self.input_proj(x)                     # [B, T, D]

        # Prepend CLS token
        cls = self.cls_token.expand(B, 1, -1)
        h = torch.cat([cls, h], dim=1)           # [B, T+1, D]
        h = self.pos_enc(h)

        for layer in self.layers:
            h = layer["attn"](h)
            h = layer["ffn"](h)
            h = layer["grn"](h)
            h = layer["se"](h)

        h = self.norm(h)
        return h[:, 0, :]                          # CLS → [B, D]


# ─────────────────────────────────────────────────────────────────────────────
# AttentionLSTM — LSTM with multi-head temporal attention
# ─────────────────────────────────────────────────────────────────────────────

class AttentionLSTMEncoder(nn.Module):
    """
    Bidirectional-compatible LSTM with multi-head attention readout.

    Architecture:
        Input projection → LSTM (multi-layer)
        → Multi-head attention over all hidden states
        → Concatenate [attention_context, last_hidden]
        → Output projection

    Hidden state (hx) is carried across steps for recurrent inference.
    """

    def __init__(self, cfg: NetworkConfig, n_features: int) -> None:
        super().__init__()
        H = cfg.lstm_hidden
        D = cfg.d_model

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, H),
            nn.LayerNorm(H),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=H,
            hidden_size=H,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            dropout=cfg.lstm_dropout if cfg.lstm_layers > 1 else 0.0,
        )
        self.lstm_norm = nn.LayerNorm(H)

        # Multi-head attention over LSTM outputs
        self.attn = nn.MultiheadAttention(
            embed_dim=H,
            num_heads=max(1, cfg.n_heads // 2),
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(H)

        # Output projection to d_model
        self.out_proj = nn.Sequential(
            nn.Linear(H * 2, D),
            nn.LayerNorm(D),
            nn.GELU(),
        )
        self.grn = GatedResidualBlock(D, D, cfg.dropout)
        
        self.out_dim = D

    def forward(
        self,
        x:  Tensor,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # x: [B, T, F]
        h = self.input_proj(x)                    # [B, T, H]

        lstm_out, hx_new = self.lstm(h, hx)       # [B, T, H]
        lstm_out = self.lstm_norm(lstm_out)

        # Self-attention over LSTM outputs
        # Query = last hidden state
        query = lstm_out[:, -1:, :]             # [B, 1, H]
        ctx, _ = self.attn(query, lstm_out, lstm_out)
        ctx = self.attn_norm(ctx.squeeze(1))  # [B, H]

        # Concatenate attention context + last LSTM output
        combined = torch.cat([ctx, lstm_out[:, -1, :]], dim=-1)  # [B, H*2]
        out = self.out_proj(combined)         # [B, D]
        out = self.grn(out)
        return out, hx_new


# ─────────────────────────────────────────────────────────────────────────────
# Temporal CNN Encoder — Dilated causal convolutions
# ─────────────────────────────────────────────────────────────────────────────

class TemporalCNNEncoder(nn.Module):
    """
    Stack of dilated causal convolutions.
    Receptive field doubles each layer: 1, 2, 4, 8, 16...
    Faster than LSTM for long sequences.

    Architecture:
        Input proj → N × DilatedCausalConv(dilation=2^i)
        → Global average pool → Output proj
    """

    def __init__(self, cfg: NetworkConfig, n_features: int) -> None:
        super().__init__()
        D = cfg.d_model

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, D),
            nn.LayerNorm(D),
            nn.GELU(),
        )

        self.conv_blocks = nn.ModuleList([
            DilatedCausalConv(
                channels=D,
                kernel=3,
                dilation=2 ** i,
                dropout=cfg.dropout,
            )
            for i in range(cfg.n_layers)
        ])

        self.se = SqueezeExcitation(D)
        self.out_dim = D
        self.norm = nn.LayerNorm(D)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, F]
        h = self.input_proj(x)
        for block in self.conv_blocks:
            h = block(h)
        h = self.se(h)
        h = self.norm(h)
        return h[:, -1, :]   # last time step → [B, D]

# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Encoder — CNN fast path + Transformer slow path
# ─────────────────────────────────────────────────────────────────────────────


class HybridEncoder(nn.Module):
    """
    Dual-path encoder:
        CNN path  → captures local micro-structure (fast)
        Attn path → captures long-range dependencies (slow)
        Fusion    → learned gated combination

    Best of both worlds: locality + global context.
    """

    def __init__(self, cfg: NetworkConfig, n_features: int) -> None:
        super().__init__()
        D = cfg.d_model

        self.cnn_path = TemporalCNNEncoder(cfg, n_features)
        self.attn_path = MarketTransformerEncoder(cfg, n_features)

        # Gated fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.Sigmoid(),
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.LayerNorm(D),
            nn.GELU(),
        )
        self.grn = GatedResidualBlock(D,D, cfg.dropout)
        self.out_dim = D

    def forward(self, x: Tensor) -> Tensor:
        cnn_out = self.cnn_path(x)              # [B, D]
        attn_out = self.attn_path(x)             # [B, D]

        combined = torch.cat([cnn_out, attn_out], dim=-1)  # [B, D*2]
        gate = self.fusion_gate(combined)    # [B, D]
        proj = self.fusion_proj(combined)    # [B, D]

        fused = gate * cnn_out + (1 - gate) * attn_out + proj
        fused = self.grn(fused)
        return fused                             # [B, D]

# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Encoder — MLP with cross-feature attention
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioEncoder(nn.Module):
    """
    Encode portfolio state vector with feature-wise attention.
    Learns which portfolio features matter most in context.
    """

    def __init__(self, n_features: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = max(out_dim, n_features * 4)

        self.proj = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        # Feature-wise attention
        self.feature_attn = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Softmax(dim=-1),
        )
        self.grn = GatedResidualBlock(out_dim,out_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, n_features]
        weights = self.feature_attn(x)     # [B, n_features]
        x_attn = x * weights              # feature reweighting
        out = self.proj(x_attn)            # [B, out_dim]
        out = self.grn(out)
        return out



def build_encoder(
    cfg:        NetworkConfig,
    n_features: int,
) -> nn.Module:
    """Factory — selects encoder from config."""
    arch = cfg.architecture
    if arch == Architecture.TRANSFORMER:
        return MarketTransformerEncoder(cfg, n_features)
    elif arch == Architecture.LSTM:
        return AttentionLSTMEncoder(cfg, n_features)
    elif arch == Architecture.CNN:
        return TemporalCNNEncoder(cfg, n_features)
    elif arch == Architecture.HYBRID:
        return HybridEncoder(cfg, n_features)
    else:
        return AttentionLSTMEncoder(cfg, n_features)



# ─────────────────────────────────────────────────────────────────────────────
# Continuous Actor (Gaussian policy)
# ─────────────────────────────────────────────────────────────────────────────

class ContinuousActor(nn.Module):
    """
    Gaussian actor for continuous action spaces.

    Outputs diagonal Gaussian: π(a|s) = N(μ(s), σ(s))
    - μ from network output
    - log σ is a learnable parameter (state-independent)
    - Actions squashed through tanh for bounded output [-1, 1]

    For SAC: uses reparameterisation trick (rsample).
    For PPO: uses sample + log_prob for importance ratio.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 0.5

    def __init__(
        self,
        trunk_dim:    int,
        action_dim:   int,
        hidden_dims:  Tuple[int, ...] = (128,),
        dropout:      float = 0.1,
        squash:       bool = False,    # tanh squashing
        state_dependent_std: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.squash = squash
        self.state_dep = state_dependent_std

        # Mean head
        layers = []
        in_dim = trunk_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.mean_net = nn.Sequential(*layers)

        # Std head
        if state_dependent_std:
            std_layers = []
            in_d = trunk_dim
            for h in hidden_dims:
                std_layers += [nn.Linear(in_d, h), nn.GELU()]
                in_d = h
            std_layers.append(nn.Linear(in_d, action_dim))
            self.log_std_net = nn.Sequential(*std_layers)
        else:
            self.log_std = nn.Parameter(
                torch.zeros(action_dim) - 0.5
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        features: Tensor,
        action:   Optional[Tensor] = None,
        deterministic: bool = False,
    ) -> ActorOutput:
        mean = self.mean_net(features)

        if self.state_dep:
            log_std = self.log_std_net(features)
        else:
            log_std = self.log_std.expand_as(mean)

        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std)

        if deterministic:
            raw_action = mean
        elif action is None:
            raw_action = dist.rsample()
        else:
            raw_action = action

        if self.squash:
            # Tanh squashing with log-prob correction
            action_out = torch.tanh(raw_action)
            # log π(a|s) = log N(u|μ,σ) - Σ log(1 - tanh²(u))
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            log_prob -= (
                2 * (math.log(2) - raw_action - F.softplus(-2 * raw_action))
            ).sum(dim=-1)
        else:
            action_out = raw_action
            log_prob = dist.log_prob(raw_action).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return ActorOutput(
            action=action_out,
            log_prob=log_prob,
            entropy=entropy,
            mean=torch.tanh(mean) if self.squash else mean,
            std=std,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Dueling Critic
# ─────────────────────────────────────────────────────────────────────────────

class DuelingCritic(nn.Module):
    """
    Dueling network architecture.
    Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]

    Value stream  → estimates how good the state is
    Advantage stream → estimates relative action quality
    Decomposition reduces variance and speeds convergence.

    Reference: Wang et al. (2016) "Dueling Network Architectures"
    """

    def __init__(
        self,
        trunk_dim:  int,
        action_dim: int,
        hidden:     int = 256,
        use_noisy: bool = True,
    ) -> None:
        super().__init__()
        Linear = NoisyLinear if use_noisy else nn.Linear
        # Value stream V(s) → scalar
        self.value_stream = nn.Sequential(
            nn.Linear(trunk_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            Linear(hidden, 1),
        )
        # Advantage stream A(s,a) → per-action
        self.adv_stream = nn.Sequential(
            nn.Linear(trunk_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            Linear(hidden, action_dim),
        )
        self.norm = nn.LayerNorm(trunk_dim)

    def forward(self, features: Tensor) -> CriticOutput:
        h = self.norm(features)
        V = self.value_stream(h)                           # [B, 1]
        A = self.adv_stream(h)                             # [B, A]
        Q = V + A - A.mean(dim=-1, keepdim=True)          # [B, A]
        return CriticOutput(value=Q)

    def get_value(self, features: Tensor) -> Tensor:
        """Return V(s) only — used for PPO critic."""
        h = self.norm(features)
        return self.value_stream(h).squeeze(-1)            # [B]

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# ─────────────────────────────────────────────────────────────────────────────
# Distributional Critic — C51 (Rainbow)
# ─────────────────────────────────────────────────────────────────────────────

class DistributionalCritic(nn.Module):
    """
    C51 distributional critic.
    Models Q-value as probability distribution over [v_min, v_max].

    Instead of Q(s,a) ∈ ℝ, output Z(s,a) ∈ Δ^{n_atoms-1}

    Advantages:
    - Captures full return distribution (not just mean)
    - More stable training through distributional Bellman
    - Natural uncertainty quantification

    Reference: Bellemare et al. (2017) "Distributional RL"
    """

    def __init__(
        self,
        trunk_dim:  int,
        action_dim: int,
        n_atoms:    int = 51,
        v_min:      float = -10.0,
        v_max:      float = 10.0,
        hidden:     int = 256,
        use_noisy:  bool = True,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        Linear = NoisyLinear if use_noisy else nn.Linear
        # Register atom support
        self.register_buffer(
            "atoms",
            torch.linspace(v_min, v_max, n_atoms),
        )
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        # Dueling distributional
        self.value_stream = nn.Sequential(
            nn.Linear(trunk_dim, hidden),
            nn.GELU(),
            Linear(hidden, n_atoms),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(trunk_dim, hidden),
            nn.GELU(),
            Linear(hidden, action_dim * n_atoms),
        )
        self.norm = nn.LayerNorm(trunk_dim)

    def forward(self, features: Tensor) -> DistributionalOutput:
        B = features.size(0)
        h = self.norm(features)

        V = self.value_stream(h).view(B, 1, self.n_atoms)
        A = self.adv_stream(h).view(B, self.action_dim, self.n_atoms)
        Q_atoms = V + A - A.mean(dim=1, keepdim=True)     # [B, A, n_atoms]

        log_probs = F.log_softmax(Q_atoms, dim=-1)
        q_values = (log_probs.exp() * self.atoms).sum(dim=-1)   # [B, A]

        return DistributionalOutput(log_probs=log_probs, q_values=q_values)

    def project_distribution(
        self,
        rewards:      Tensor,
        next_log_probs: Tensor,
        dones:        Tensor,
        gamma:        float,
    ) -> Tensor:
        """
        Distributional Bellman projection (L2 projection).
        Projects target distribution onto support atoms.

        Returns target log-probs [B, A, n_atoms].
        """
        """Vectorized distributional Bellman projection."""
        B = rewards.size(0)
        N = self.n_atoms
        atoms = self.atoms

        # Projected target atoms: r + γ * (1-done) * z
        target = (
            rewards.unsqueeze(-1)
            + gamma * (1 - dones.unsqueeze(-1)) * atoms
        ).clamp(self.v_min, self.v_max)  # [B, N]

        # Map to atom indices
        b = (target - self.v_min) / self.delta_z  # [B, N]
        l = b.floor().long().clamp(0, N - 1)      # [B, N]
        u = b.ceil().long().clamp(0, N - 1)       # [B, N]

        # Vectorized scatter (no Python loop)
        m = torch.zeros(B, N, device=rewards.device)
        next_probs = next_log_probs.exp()  # [B, N]

        # Create batch indices for 2D scatter
        batch_idx = torch.arange(
            B, device=rewards.device).unsqueeze(1).expand(-1, N)

        # Lower bound contribution
        m[batch_idx, l] += next_probs * (1 - (b - l.float()))
        # Upper bound contribution
        m[batch_idx, u] += next_probs * (b - l.float())

        return m.clamp(1e-8, 1.0).log()

    def reset_noise(self) -> None:
        for mod in self.modules():
            if isinstance(mod, NoisyLinear):
                mod.reset_noise()


# ─────────────────────────────────────────────────────────────────────────────
# Twin Q-Network Critic (for SAC / TD3)
# ─────────────────────────────────────────────────────────────────────────────

class TwinQCritic(nn.Module):
    """
    Standard Q-network for continuous action spaces.
    Used as one of two twin critics in SAC and TD3.

    Takes state features and action as input, outputs scalar Q-value.
    Q(s, a) → ℝ

    Architecture:
        [state_features ; action] → MLP → Q-value

    Args:
        state_dim:     Dimension of state encoding (trunk output)
        action_dim:    Dimension of action space
        hidden_dims:   Hidden layer sizes
    """

    def __init__(
        self,
        state_dim:   int,
        action_dim:  int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()

        # Build MLP: [state ; action] → hidden → ... → 1
        layers: List[nn.Module] = []
        input_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal initialization with ReLU gain."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        """
        Args:
            state:  Encoded state features [B, state_dim]
            action: Action tensor [B, action_dim]
        Returns:
            Q-value [B, 1]
        """
        combined = torch.cat([state, action], dim=-1)
        return self.network(combined)


# ─────────────────────────────────────────────────────────────────────────────
# Discrete Actor (Categorical policy)
# ─────────────────────────────────────────────────────────────────────────────

class DiscreteActor(nn.Module):
    """
    Discrete action actor using Categorical distribution.
    Used for environments with a finite set of actions (e.g., 6 trading actions).

    Outputs logits over actions, sampled via Categorical distribution.
    π(a|s) = Categorical(logits = MLP(state))

    Architecture:
        State features → MLP → Logits → Categorical → Action

    Args:
        state_dim:    Dimension of state encoding (trunk output)
        num_actions:  Number of discrete actions
        hidden_dims:  Hidden layer sizes
        dropout:      Dropout rate between hidden layers
    """

    def __init__(
        self,
        state_dim:    int,
        num_actions:  int,
        hidden_dims:  Tuple[int, ...] = (128,),
        dropout:      float = 0.1,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions

        # Build MLP: state → hidden → ... → logits
        layers: List[nn.Module] = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, num_actions))
        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal initialization with small gain for policy head."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        state_features: Tensor,
        action:         Optional[Tensor] = None,
        deterministic:  bool = False,
    ) -> ActorOutput:
        """
        Args:
            state_features: Encoded state [B, state_dim]
            action:         Pre-selected action (for importance sampling)
            deterministic:  If True, take argmax instead of sampling

        Returns:
            ActorOutput with action, log_prob, entropy, mean (logits), std (zeros)
        """
        logits = self.network(
            state_features)                      # [B, num_actions]
        distribution = torch.distributions.Categorical(logits=logits)

        # Select action
        if deterministic:
            selected_action = logits.argmax(dim=-1)
        elif action is not None:
            selected_action = action
        else:
            selected_action = distribution.sample()

        return ActorOutput(
            action=selected_action,
            log_prob=distribution.log_prob(selected_action),
            entropy=distribution.entropy(),
            mean=logits,                                        # Logits as "mean"
            # No std for discrete
            std=torch.zeros_like(logits),
        )
