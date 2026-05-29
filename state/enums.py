
from __future__ import annotations


from enum import Enum

class Side(str, Enum):
    """Position direction."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class AssetClass(str, Enum):
    """Tradable asset classes."""
    EQUITY = "equity"
    CRYPTO = "crypto"
    FOREX = "forex"
    FUTURES = "futures"
    OPTIONS = "options"


class TimeFrame(str, Enum):
    """Bar timeframes."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class EpisodeTermination(str, Enum):
    """Causes of episode termination."""
    MAX_STEPS = "max_steps"
    MAX_DRAWDOWN = "max_drawdown"
    MIN_CAPITAL = "min_capital"
    DATA_EXHAUSTED = "data_exhausted"
    AGENT_HALT = "agent_halt"


class DRLAlgorithm(str, Enum):
    """Available RL algorithms."""
    PPO = "ppo"
    SAC = "sac"
    TD3 = "td3"
    RAINBOW_DQN = "rainbow_dqn"
    RECURRENT_PPO = "recurrent_ppo"


class Architecture(str, Enum):
    """Available network architectures."""
    MLP = "mlp"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    ATTENTION_LSTM = "attention_lstm"
    CNN = "cnn"
    HYBRID = "hybrid"


class RewardScheme(str, Enum):
    """Available reward computation schemes."""
    SHARPE = "sharpe"
    SORTINO = "sortino"
    CALMAR = "calmar"
    RISK_ADJUSTED = "risk_adjusted"
    ASYMMETRIC = "asymmetric"
    PROFIT_FACTOR = "profit_factor"
    RAW_PNL = "raw_pnl"


class ActionSpace:
    """Action space type constants."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


class MarketRegime(Enum):
    CRASH = -2
    BEAR = -1
    SIDEWAYS = 0
    BULL = 1
    BUBBLE = 2
