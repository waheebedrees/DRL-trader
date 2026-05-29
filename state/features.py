# state/features.py
"""
Single source of truth for all feature definitions.
Every dimension has a name — critical for debugging and interpretability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class FeatureGroup:
    name:     str
    features: Tuple[str, ...]

    @property
    def dim(self) -> int:
        return len(self.features)


# ── Market sequence features (per bar) ────────────────────────────────────────

PRICE_GROUP = FeatureGroup("price", (
    "log_return",
    "high_low_range",
    "close_position",
    "open_gap",
))

VOLUME_GROUP = FeatureGroup("volume", (
    "volume_ratio",
    "volume_momentum",
    "volume_surprise",
))

MOMENTUM_GROUP = FeatureGroup("momentum", (
    "rsi_norm",
    "macd_norm",
    "macd_hist_norm",
    "stoch_k_norm",
    "stoch_d_norm",
    "williams_r_norm",
    "cci_norm",
    "mfi_norm",
))

TREND_GROUP = FeatureGroup("trend", (
    "adx_norm",
    "adx_plus_di_norm",
    "adx_minus_di_norm",
    "ema_12_dist",
    "ema_26_dist",
    "sma_50_dist",
    "sma_200_dist",
))

VOLATILITY_GROUP = FeatureGroup("volatility", (
    "atr_norm",
    "bb_pct",
    "bb_width_norm",
    "realised_vol",
))

VOLUME_INDICATORS_GROUP = FeatureGroup("volume_indicators", (
    "cmf",
    "mfi_norm_2",
    "obv_momentum",
    "vwap_dist",
))

MICROSTRUCTURE_GROUP = FeatureGroup("microstructure", (
    "spread_bps",
    "book_imbalance",
    "depth_ratio",
))

REGIME_GROUP = FeatureGroup("regime", (
    "trend_strength",
    "vol_regime",
    "momentum_composite",
))

# All market sequence groups in order
MARKET_GROUPS: Tuple[FeatureGroup, ...] = (
    PRICE_GROUP,
    VOLUME_GROUP,
    MOMENTUM_GROUP,
    TREND_GROUP,
    VOLATILITY_GROUP,
    VOLUME_INDICATORS_GROUP,
    MICROSTRUCTURE_GROUP,
    REGIME_GROUP,
)

N_MARKET_FEATURES: int = sum(g.dim for g in MARKET_GROUPS)

MARKET_FEATURE_NAMES: Tuple[str, ...] = tuple(
    f for g in MARKET_GROUPS for f in g.features
)

# ── Portfolio vector features ─────────────────────────────────────────────────

PORTFOLIO_GROUP = FeatureGroup("portfolio", (
    "position_side",       # -1, 0, +1
    "position_size_norm",  # notional / capital
    "unrealised_pnl_norm",  # pnl / capital
    "drawdown_norm",
    "cash_ratio",          # cash / capital
    "exposure_norm",
    "portfolio_heat",
    "hold_bars_norm",
    "dist_to_sl",
    "dist_to_tp",
))

N_PORTFOLIO_FEATURES: int = PORTFOLIO_GROUP.dim

# ── Sentiment vector features ─────────────────────────────────────────────────

SENTIMENT_GROUP = FeatureGroup("sentiment", (
    "overall_score",
    "confidence",
    "news_score",
    "social_score",
    "options_flow_score",
))

N_SENTIMENT_FEATURES: int = SENTIMENT_GROUP.dim

# ── Time encoding features ────────────────────────────────────────────────────

TIME_GROUP = FeatureGroup("time", (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
))

N_TIME_FEATURES: int = TIME_GROUP.dim

# ── Flat observation index map ────────────────────────────────────────────────


def build_obs_index_map(lookback: int) -> dict:
    """
    Returns dict mapping feature_name → (start_idx, end_idx)
    for the flat observation vector.
    Useful for debugging specific feature values.
    """
    idx_map = {}
    cursor = 0
    seq_dim = lookback * N_MARKET_FEATURES

    # Market sequence (flat)
    idx_map["market_sequence"] = (cursor, cursor + seq_dim)
    cursor += seq_dim

    # Portfolio
    idx_map["portfolio"] = (cursor, cursor + N_PORTFOLIO_FEATURES)
    for i, name in enumerate(PORTFOLIO_GROUP.features):
        idx_map[f"portfolio.{name}"] = (cursor + i, cursor + i + 1)
    cursor += N_PORTFOLIO_FEATURES

    # Sentiment
    idx_map["sentiment"] = (cursor, cursor + N_SENTIMENT_FEATURES)
    for i, name in enumerate(SENTIMENT_GROUP.features):
        idx_map[f"sentiment.{name}"] = (cursor + i, cursor + i + 1)
    cursor += N_SENTIMENT_FEATURES

    # Time
    idx_map["time"] = (cursor, cursor + N_TIME_FEATURES)
    for i, name in enumerate(TIME_GROUP.features):
        idx_map[f"time.{name}"] = (cursor + i, cursor + i + 1)
    cursor += N_TIME_FEATURES

    idx_map["_total_dim"] = cursor
    return idx_map


FLAT_OBS_DIM_BASE = N_PORTFOLIO_FEATURES + \
    N_SENTIMENT_FEATURES + N_TIME_FEATURES


def flat_obs_dim(lookback: int = 60) -> int:
    return lookback * N_MARKET_FEATURES + FLAT_OBS_DIM_BASE


MARKET_FEATURE_NAMES: Tuple[str, ...] = (
    "log_return",       "hl_range",         "close_position",  "open_gap",
    "volume_ratio",     "volume_momentum",  "volume_raw_norm",
    "rsi_norm",         "macd_norm",        "macd_hist_norm",
    "stoch_k_norm",     "stoch_d_norm",     "williams_r_norm",
    "cci_norm",         "mfi_norm",
    "adx_norm",         "adx_plus_di",      "adx_minus_di",
    "ema12_dist",       "ema26_dist",       "sma50_dist",      "sma200_dist",
    "realised_vol",     "atr_norm",         "bb_pct",
    "bb_width_norm",    "keltner_width",    "cmf",
    "obv_momentum",     "vwap_dist",        "adl_norm",
    "spread_bps",       "book_imbalance",   "depth_ratio",
    "trend_strength",   "vol_regime",       "momentum_composite",
)
N_MARKET_FEATURES:    int = len(MARKET_FEATURE_NAMES)   # 37
N_PORTFOLIO_FEATURES: int = 10
N_SENTIMENT_FEATURES: int = 5
N_TIME_FEATURES:      int = 6
