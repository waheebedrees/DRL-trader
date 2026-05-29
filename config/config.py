from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from state import Architecture, ActionSpace, DRLAlgorithm, RewardScheme


@dataclass
class NetworkConfig:
    architecture:  Architecture = Architecture.ATTENTION_LSTM
    hidden_dims:   Tuple[int, ...] = (256, 128)
    activation:    str = "gelu"
    layer_norm:    bool = True
    dropout:       float = 0.2          # CHANGED: was 0.1, raised for regularization

    # LSTM
    lstm_hidden:   int = 128
    lstm_layers:   int = 1
    lstm_dropout:  float = 0.1          # CHANGED: was 0.0

    # Transformer
    d_model:       int = 128
    n_heads:       int = 4
    n_layers:      int = 2
    d_ff:          int = 256
    max_seq_len:   int = 256

    # Extras
    use_dueling:   bool = True
    use_noisy_net: bool = True
    weight_init:   str = "orthogonal"


# ══════════════════════════════════════════════════════
# SECTION: PPO Config
# ══════════════════════════════════════════════════════

@dataclass
class PPOConfig:
    clip_epsilon:         float = 0.2
    clip_vf_epsilon:      float = 0.2
    entropy_coef:         float = 0.02         # CHANGED: was 0.01
    entropy_coef_min:     float = 0.005        # CHANGED: was 0.001
    entropy_anneal_steps: int = 500_000      # CHANGED: was 200_000
    value_coef:           float = 0.5
    max_grad_norm:        float = 1.0          # CHANGED: was 0.5
    gae_lambda:           float = 0.95
    gamma:                float = 0.99
    n_steps:              int = 1024         # CHANGED: was 512
    n_epochs:             int = 6            # CHANGED: was 4
    batch_size:           int = 128          # CHANGED: was 64
    learning_rate:        float = 2e-4         # CHANGED: was 3e-4
    warmup_steps:         int = 256          # CHANGED: was 128
    normalize_advantage:  bool = True
    normalize_returns:    bool = True
    lr_schedule:          str = "linear"
    target_kl:            Optional[float] = 0.10   # CHANGED: was 0.05
    use_amp:              bool = False


# ══════════════════════════════════════════════════════
# SECTION: SAC Config  (unchanged)
# ══════════════════════════════════════════════════════

@dataclass
class SACConfig:
    alpha:           float = 0.2
    auto_alpha:      bool = True
    target_entropy:  Optional[float] = None
    gamma:           float = 0.99
    tau:             float = 0.005
    batch_size:      int = 64
    buffer_size:     int = 100_000
    learning_rate:   float = 3e-4
    learning_starts: int = 10
    train_freq:      int = 1
    gradient_steps:  int = 1
    max_grad_norm:   float = 1.0


# ══════════════════════════════════════════════════════
# SECTION: TD3 Config  (unchanged)
# ══════════════════════════════════════════════════════

@dataclass
class TD3Config:
    gamma:             float = 0.99
    tau:               float = 0.005
    policy_noise:      float = 0.2
    noise_clip:        float = 0.5
    policy_delay:      int = 2
    batch_size:        int = 64
    buffer_size:       int = 100_000
    learning_rate:     float = 3e-4
    learning_starts:   int = 10
    exploration_noise: float = 0.1
    max_grad_norm:     float = 1.0


# ══════════════════════════════════════════════════════
# SECTION: Rainbow Config  (unchanged)
# ══════════════════════════════════════════════════════

@dataclass
class RainbowConfig:
    gamma:           float = 0.99
    per_alpha:       float = 0.6
    per_beta:        float = 0.4
    per_beta_frames: int = 100_000
    per_epsilon:     float = 1e-6
    n_atoms:         int = 51
    v_min:           float = -10.0
    v_max:           float = 10.0
    batch_size:      int = 32
    buffer_size:     int = 100_000
    learning_rate:   float = 6.25e-5
    target_update:   int = 2_000
    max_grad_norm:   float = 10.0


# ══════════════════════════════════════════════════════
# SECTION: Reward Config  ← KEY CHANGES HERE
# ══════════════════════════════════════════════════════

@dataclass
class RewardConfig:
    returns_window:  int = 100
    risk_free_rate:  float = 0.04

    # Component weights
    w_pnl:           float = 0.65   # CHANGED: was 0.60
    w_sharpe:        float = 0.10
    w_drawdown:      float = 0.15
    w_trade_quality: float = 0.10

    # Bonuses
    bonus_hit_tp:            float = 0.15   # CHANGED: was 0.20
    bonus_correct_direction: float = 0.02
    bonus_hold_profit:       float = 0.01   # CHANGED: was 0.005

    # Penalties  ← CRITICAL CHANGES
    penalty_hit_sl:         float = -0.05   # CHANGED: was -0.10 (SL is good!)
    penalty_overtrade:      float = -0.05
    penalty_large_drawdown: float = -0.25
    penalty_hold_too_long:  float = -0.005
    penalty_over_leverage:  float = -0.15

    # Thresholds  ← CRITICAL CHANGES
    hold_profit_threshold:  float = 0.002   # CHANGED: was 0.001
    max_drawdown_threshold: float = 0.10    # CHANGED: was 0.10 (keep)
    max_daily_trades:       int = 25      # CHANGED: was 20
    max_hold_bars:          int = 150
    max_leverage:           float = 3.0

    # Scaling  ← CRITICAL CHANGE
    reward_scale:           float = 1.0     # CHANGED: was 0.1 (too small!)
    clip_reward:            float = 5.0     # CHANGED: was 2.0


# ══════════════════════════════════════════════════════
# SECTION: Environment Config
# ══════════════════════════════════════════════════════

@dataclass
class EnvironmentConfig:
    episode_length:         int = 400     # CHANGED: was 300
    warmup_bars:            int = 60
    initial_capital:        float = 100_000.0
    action_space:           str = ActionSpace.CONTINUOUS
    n_discrete_actions:     int = 6
    commission_rate:        float = 0.001
    slippage_rate:          float = 0.0005
    funding_rate_per_bar:   float = 0.00005
    max_leverage:           float = 3.0
    max_drawdown_terminate: float = 0.30    # CHANGED: was 0.25
    min_capital_pct:        float = 0.50
    reward: RewardConfig = field(default_factory=RewardConfig)
    lookback_window:        int = 60
    normalise_obs:          bool = True
    clip_obs:               float = 5.0
    random_start:           bool = True
    include_orderbook:      bool = True
    include_sentiment:      bool = True


# ══════════════════════════════════════════════════════
# SECTION: Training Config  (unchanged)
# ══════════════════════════════════════════════════════

@dataclass(frozen=True)
class TrainingConfig:
    algorithm:      DRLAlgorithm = DRLAlgorithm.PPO
    reward_scheme:  RewardScheme = RewardScheme.RISK_ADJUSTED
    network:        NetworkConfig = field(default_factory=NetworkConfig)
    environment:    EnvironmentConfig = field(
        default_factory=EnvironmentConfig)
    ppo:            PPOConfig = field(default_factory=PPOConfig)
    sac:            SACConfig = field(default_factory=SACConfig)
    td3:            TD3Config = field(default_factory=TD3Config)
    rainbow:        RainbowConfig = field(default_factory=RainbowConfig)
    total_timesteps: int = 1_000_000
    eval_freq:       int = 10_000
    eval_episodes:   int = 20
    save_freq:       int = 50_000
    log_interval:    int = 100
    n_envs:          int = 4
    seed:            int = 42
    device:          str = "auto"
    model_dir:       str = "models/drl"


# ══════════════════════════════════════════════════════
# SECTION: Market Config  (unchanged)
# ══════════════════════════════════════════════════════

@dataclass
class MarketConfig:
    initial_price:          float = 100.0
    base_volume:            float = 1_000_000
    min_price:              float = 1.0
    max_price:              float = 1_000_000.0
    max_return:             float = 0.15
    volatility_base:        float = 0.02
    volatility_persistence: float = 0.85
    regime_change_prob:     float = 0.005
    fat_tail_prob:          float = 0.02
    microstructure_noise:   float = 0.0005
    regime_transitions:     np.ndarray = None

    def __post_init__(self):
        if self.regime_transitions is None:
            self.regime_transitions = np.array([
                [0.2, 0.4, 0.3, 0.1, 0.0],
                [0.1, 0.4, 0.3, 0.2, 0.0],
                [0.05, 0.15, 0.5, 0.25, 0.05],
                [0.0, 0.1, 0.3, 0.4, 0.2],
                [0.3, 0.3, 0.2, 0.1, 0.1],
            ])


# config.py - REPLACE these dataclasses entirely

@dataclass
class PPOConfig:
    clip_epsilon:         float = 0.2
    clip_vf_epsilon:      float = 0.2

    # CRITICAL: entropy was 0.02, causing exploration chaos
    # Start high, anneal fast so policy converges in first 50 iters
    entropy_coef:         float = 0.005   # was 0.02  ← TOO HIGH
    entropy_coef_min:     float = 0.001
    entropy_anneal_steps: int = 100_000  # was 500_000 ← anneals too slow

    value_coef:           float = 0.5
    max_grad_norm:        float = 0.5     # was 1.0 ← too permissive
    gae_lambda:           float = 0.95
    gamma:                float = 0.99
    n_steps:              int = 512     # was 1024 ← too long on CPU
    n_epochs:             int = 4
    batch_size:           int = 64
    learning_rate:        float = 1e-4   # was 2e-4 ← too fast with random actions
    warmup_steps:         int = 512
    normalize_advantage:  bool = True
    normalize_returns:    bool = False  # was True ← distorts sparse rewards
    lr_schedule:          str = "linear"
    target_kl:            Optional[float] = 0.02  # was 0.10 ← too permissive
    use_amp:              bool = False


@dataclass
class RewardConfig:
    returns_window:  int = 100
    risk_free_rate:  float = 0.04

    # Weights — PnL must dominate during early training
    w_pnl:           float = 0.80   # was 0.65 ← increase signal
    w_sharpe:        float = 0.05   # was 0.10 ← reduce noise
    w_drawdown:      float = 0.10   # was 0.15 ← reduce suppression
    w_trade_quality: float = 0.05   # was 0.10

    # Bonuses
    bonus_hit_tp:            float = 0.10
    bonus_correct_direction: float = 0.01
    bonus_hold_profit:       float = 0.005

    # Penalties — MUCH softer so they don't drown PnL
    penalty_hit_sl:          float = -0.02  # was -0.05
    penalty_overtrade:       float = -0.02  # was -0.05
    penalty_large_drawdown:  float = -0.10  # was -0.25 ← was killing signal
    penalty_hold_too_long:   float = -0.002
    penalty_over_leverage:   float = -0.05

    # Thresholds — give agent more room
    hold_profit_threshold:   float = 0.001
    max_drawdown_threshold:  float = 0.15  # was 0.10 ← fires too often
    max_daily_trades:        int = 50    # was 25 ← too restrictive early
    max_hold_bars:           int = 200
    max_leverage:            float = 3.0

    # Scaling
    reward_scale:            float = 1.0
    clip_reward:             float = 3.0   # was 5.0


@dataclass
class EnvironmentConfig:
    # was 400 ← shorter = more resets = more learning signal
    episode_length:         int = 200
    warmup_bars:            int = 60
    initial_capital:        float = 100_000.0
    action_space:           str = ActionSpace.CONTINUOUS
    n_discrete_actions:     int = 6
    commission_rate:        float = 0.001
    slippage_rate:          float = 0.0005
    funding_rate_per_bar:   float = 0.00005
    max_leverage:           float = 3.0
    max_drawdown_terminate: float = 0.40  # was 0.30 ← terminates too early
    min_capital_pct:        float = 0.50
    reward:                 RewardConfig = field(default_factory=RewardConfig)
    lookback_window:        int = 60
    normalise_obs:          bool = True
    clip_obs:               float = 5.0
    random_start:           bool = True
    include_orderbook:      bool = True
    include_sentiment:      bool = True


# ── Network ───────────────────────────────────────────────────────────────────

@dataclass
class NetworkConfig:
    architecture:  Architecture = Architecture.ATTENTION_LSTM
    hidden_dims:   Tuple[int, ...] = (128, 64)
    activation:    str = "gelu"
    layer_norm:    bool = True
    dropout:       float = 0.10

    lstm_hidden:   int = 64
    lstm_layers:   int = 1
    lstm_dropout:  float = 0.0

    d_model:       int = 64
    n_heads:       int = 4
    n_layers:      int = 2
    d_ff:          int = 128
    max_seq_len:   int = 256

    use_dueling:   bool = True
    use_noisy_net: bool = True
    weight_init:   str = "orthogonal"


# ── PPO ───────────────────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    clip_epsilon:         float = 0.2
    clip_vf_epsilon:      float = 0.2
    entropy_coef:         float = 0.01    # start moderate
    entropy_coef_min:     float = 0.001
    entropy_anneal_steps: int = 200_000
    value_coef:           float = 0.5
    max_grad_norm:        float = 0.5
    gae_lambda:           float = 0.95
    gamma:                float = 0.99
    n_steps:              int = 512
    n_epochs:             int = 4
    batch_size:           int = 64
    learning_rate:        float = 3e-4
    warmup_steps:         int = 256
    normalize_advantage:  bool = True
    normalize_returns:    bool = False   # False = less distortion
    lr_schedule:          str = "linear"
    target_kl:            Optional[float] = 0.05
    use_amp:              bool = False


# ── SAC ───────────────────────────────────────────────────────────────────────

@dataclass
class SACConfig:
    alpha:           float = 0.2
    auto_alpha:      bool = True
    target_entropy:  Optional[float] = None
    gamma:           float = 0.99
    tau:             float = 0.005
    batch_size:      int = 64
    buffer_size:     int = 100_000
    learning_rate:   float = 3e-4
    learning_starts: int = 10
    train_freq:      int = 1
    gradient_steps:  int = 1
    max_grad_norm:   float = 1.0


# ── TD3 ───────────────────────────────────────────────────────────────────────

@dataclass
class TD3Config:
    gamma:             float = 0.99
    tau:               float = 0.005
    policy_noise:      float = 0.2
    noise_clip:        float = 0.5
    policy_delay:      int = 2
    batch_size:        int = 64
    buffer_size:       int = 100_000
    learning_rate:     float = 3e-4
    learning_starts:   int = 10
    exploration_noise: float = 0.1
    max_grad_norm:     float = 1.0


# ── Rainbow ───────────────────────────────────────────────────────────────────

@dataclass
class RainbowConfig:
    gamma:           float = 0.99
    per_alpha:       float = 0.6
    per_beta:        float = 0.4
    per_beta_frames: int = 100_000
    per_epsilon:     float = 1e-6
    n_atoms:         int = 51
    v_min:           float = -10.0
    v_max:           float = 10.0
    batch_size:      int = 32
    buffer_size:     int = 100_000
    learning_rate:   float = 6.25e-5
    target_update:   int = 2_000
    max_grad_norm:   float = 10.0


# ── Reward ────────────────────────────────────────────────────────────────────

@dataclass
class RewardConfig:
    returns_window:  int = 100
    risk_free_rate:  float = 0.04

    # Weights — PnL must be dominant signal
    w_pnl:           float = 0.70
    w_sharpe:        float = 0.10
    w_drawdown:      float = 0.10
    w_trade_quality: float = 0.10

    # Bonuses
    bonus_hit_tp:            float = 0.10
    bonus_correct_direction: float = 0.01
    bonus_hold_profit:       float = 0.005

    # Penalties — soft so they don't drown PnL signal
    penalty_hit_sl:          float = -0.02
    penalty_overtrade:       float = -0.02
    penalty_large_drawdown:  float = -0.15
    penalty_hold_too_long:   float = -0.001
    penalty_over_leverage:   float = -0.05

    # Thresholds — generous so penalties fire rarely
    hold_profit_threshold:   float = 0.001
    max_drawdown_threshold:  float = 0.15   # only penalise >15% DD
    max_daily_trades:        int = 50
    max_hold_bars:           int = 200
    max_leverage:            float = 3.0

    # Scaling
    reward_scale:            float = 1.0
    clip_reward:             float = 3.0


# ── Environment ───────────────────────────────────────────────────────────────

@dataclass
class EnvironmentConfig:
    episode_length:         int = 200
    warmup_bars:            int = 60
    initial_capital:        float = 100_000.0
    action_space:           str = ActionSpace.CONTINUOUS
    n_discrete_actions:     int = 6
    commission_rate:        float = 0.001
    slippage_rate:          float = 0.0005
    funding_rate_per_bar:   float = 0.00005
    max_leverage:           float = 3.0
    max_drawdown_terminate: float = 0.40
    min_capital_pct:        float = 0.40
    reward:  RewardConfig = field(default_factory=RewardConfig)
    lookback_window:        int = 60
    normalise_obs:          bool = True
    clip_obs:               float = 5.0
    random_start:           bool = True
    include_orderbook:      bool = True
    include_sentiment:      bool = True


# ── Training ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrainingConfig:
    algorithm:       DRLAlgorithm = DRLAlgorithm.PPO
    reward_scheme:   RewardScheme = RewardScheme.RISK_ADJUSTED
    network:         NetworkConfig = field(default_factory=NetworkConfig)
    environment:     EnvironmentConfig = field(
        default_factory=EnvironmentConfig)
    ppo:             PPOConfig = field(default_factory=PPOConfig)
    sac:             SACConfig = field(default_factory=SACConfig)
    td3:             TD3Config = field(default_factory=TD3Config)
    rainbow:         RainbowConfig = field(default_factory=RainbowConfig)
    total_timesteps: int = 1_000_000
    eval_freq:       int = 10_000
    eval_episodes:   int = 20
    save_freq:       int = 50_000
    log_interval:    int = 100
    n_envs:          int = 1
    seed:            int = 42
    device:          str = "auto"
    model_dir:       str = "models/drl"


# ── Market ────────────────────────────────────────────────────────────────────

@dataclass
class MarketConfig:
    initial_price:          float = 100.0
    base_volume:            float = 1_000_000
    min_price:              float = 1.0
    max_price:              float = 1_000_000.0
    max_return:             float = 0.15
    volatility_base:        float = 0.02
    volatility_persistence: float = 0.85
    regime_change_prob:     float = 0.005
    fat_tail_prob:          float = 0.02
    microstructure_noise:   float = 0.0005
    regime_transitions:     np.ndarray = None

    def __post_init__(self):
        if self.regime_transitions is None:
            self.regime_transitions = np.array([
                [0.2, 0.4, 0.3, 0.1, 0.0],
                [0.1, 0.4, 0.3, 0.2, 0.0],
                [0.05, 0.15, 0.5, 0.25, 0.05],
                [0.0, 0.1, 0.3, 0.4, 0.2],
                [0.3, 0.3, 0.2, 0.1, 0.1],
            ])
