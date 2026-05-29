from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from state import Architecture, ActionSpace, DRLAlgorithm, RewardScheme


@dataclass
class PPOConfig:
    clip_epsilon:         float = 0.2
    clip_vf_epsilon:      float = 0.2
    entropy_coef:         float = 0.01
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
    warmup_steps:         int = 128
    normalize_advantage:  bool = True
    normalize_returns:    bool = True
    lr_schedule:         str = "linear"

    # R-04: raised from 0.02 — allows policy to move more in early training
    target_kl:            Optional[float] = 0.05
    use_amp:              bool = False


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


@dataclass
class NetworkConfig:
    architecture:  Architecture = Architecture.ATTENTION_LSTM
    hidden_dims:   Tuple[int, ...] = (256, 128)
    activation:     str = "gelu"
    layer_norm:     bool = True
    dropout:       float = 0.1
    

    # LSTM
    lstm_hidden:   int = 128
    lstm_layers:   int = 1
    lstm_dropout:  float = 0.0
    
    # Transformer
    d_model:       int = 128
    n_heads:       int = 4
    n_layers:      int = 2
    d_ff:          int = 256
    max_seq_len:   int = 256
    
    # Extras
    use_dueling:   bool = True
    use_noisy_net: bool = True
    weight_init:    str = "orthogonal"


@dataclass
class RewardConfig:
    returns_window:          int = 100
    risk_free_rate:          float = 0.04
    
    
    # Component weights
    # Component weights — must reflect what we actually want the agent to optimise
    w_pnl:                   float = 0.60   # realised trade PnL (dominant)
    w_sharpe:                float = 0.15   # rolling Sharpe of total returns
    w_drawdown:              float = 0.15   # drawdown penalty
    w_trade_quality:         float = 0.10   # win-rate based quality
    
    # Bonuses
    # R-01: risk_compliance weight removed — was giving free reward for doing nothing
    # Bonuses and penalties (added on top of weighted terms)
    bonus_hit_tp:            float = 0.20
    bonus_correct_direction: float = 0.02
    # R-06: small per-bar bonus for holding a currently profitable position
    bonus_hold_profit:       float = 0.005
    
    # Penalties
    # unrealised_pnl/capital must exceed this
    penalty_hit_sl:          float = -0.10
    penalty_overtrade:       float = -0.05
    penalty_large_drawdown:  float = -0.25
    penalty_hold_too_long:   float = -0.005
    penalty_over_leverage:   float = -0.15
    
    # Thresholds
    hold_profit_threshold:   float = 0.001
    max_drawdown_threshold:  float = 0.10
    max_daily_trades:        int = 20
    max_hold_bars:           int = 150
    max_leverage:            float = 3.0
    
    # Scaling
    # R-05: single scale at the end; all components are in natural units
    reward_scale:            float = 0.1
    clip_reward:             float = 2.0


@dataclass
class EnvironmentConfig:
    # Episode
    episode_length:         int = 300
    warmup_bars:            int = 60
    initial_capital:        float = 100_000.0
    
    # Action space
    action_space:           str = ActionSpace.CONTINUOUS
    n_discrete_actions:     int = 6
    
    # Execution costs
    commission_rate:        float = 0.001
    slippage_rate:          float = 0.0005
    funding_rate_per_bar:   float = 0.00005
    max_leverage:           float = 3.0
    
    # Termination
    max_drawdown_terminate: float = 0.25
    min_capital_pct:        float = 0.50
    
    # Reward
    reward: RewardConfig = field(default_factory=RewardConfig)

    # Feature
    lookback_window:        int = 60
    normalise_obs:          bool = True
    clip_obs:               float = 5.0
    random_start:           bool = True
    include_orderbook:       bool = True
    include_sentiment:       bool = True
    
    
@dataclass(frozen=True)
class TrainingConfig:
    algorithm:       DRLAlgorithm = DRLAlgorithm.PPO
    reward_scheme:   RewardScheme = RewardScheme.RISK_ADJUSTED
    network:         NetworkConfig = field(default_factory=NetworkConfig)
    environment:     EnvironmentConfig = field(
        default_factory=EnvironmentConfig)

    ppo:     PPOConfig = field(default_factory=PPOConfig)
    sac:     SACConfig = field(default_factory=SACConfig)
    td3:     TD3Config = field(default_factory=TD3Config)
    rainbow: RainbowConfig = field(default_factory=RainbowConfig)

    total_timesteps:  int = 1_000_000
    eval_freq:        int = 10_000
    eval_episodes:    int = 20
    save_freq:        int = 50_000
    log_interval:     int = 100
    n_envs:           int = 4
    seed:             int = 42
    device:           str = "auto"
    model_dir:        str = "models/drl"


@dataclass
class MarketConfig:
    """Configuration for realistic market simulation"""
    initial_price: float = 100.0
    base_volume: float = 1_000_000
    min_price: float = 1.0
    max_price: float = 1_000_000.0
    max_return: float = 0.15  # Max ±15% per bar
    volatility_base: float = 0.02
    volatility_persistence: float = 0.85  # GARCH parameter
    regime_change_prob: float = 0.005  # Per-step probability
    fat_tail_prob: float = 0.02  # 2% chance of fat tail event
    microstructure_noise: float = 0.0005  # Bid-ask spread base

    # Regime transition matrix (row = from, col = to)
    regime_transitions: np.ndarray = None

    def __post_init__(self):
        if self.regime_transitions is None:
            # Default transition matrix
            # Order: CRASH, BEAR, SIDEWAYS, BULL, BUBBLE
            self.regime_transitions = np.array([
                [0.2, 0.4, 0.3, 0.1, 0.0],  # From CRASH
                [0.1, 0.4, 0.3, 0.2, 0.0],  # From BEAR
                [0.05, 0.15, 0.5, 0.25, 0.05],  # From SIDEWAYS
                [0.0, 0.1, 0.3, 0.4, 0.2],  # From BULL
                [0.3, 0.3, 0.2, 0.1, 0.1],  # From BUBBLE
            ])
