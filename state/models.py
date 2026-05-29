from __future__ import annotations

import copy
import math
import os
import random
import uuid
import datetime
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, NamedTuple, Optional, Tuple
from torch import Tensor

from state.enums import TimeFrame, Side, EpisodeTermination

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Core Types
# ═══════════════════════════════════════════════════════════════════════════════


class OrderBookLevel(NamedTuple):
    """Single order book price level."""
    price: float
    size:  float


class Bar(NamedTuple):
    """
    Immutable OHLCV bar. NamedTuple for performance in tight loops.

    Attributes:
        symbol: Trading pair identifier (e.g. 'BTC/USDT')
        timeframe: Bar timeframe
        timestamp: Unix timestamp in seconds
        open, high, low, close: OHLC prices
        volume: Trade volume in base currency
        vwap: Volume-weighted average price (optional)
        trades: Number of trades in this bar (optional)
    """
    symbol:    str
    timeframe: TimeFrame
    timestamp: float
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float
    vwap:      float = 0.0
    trades:    int = 0

    @property
    def bar_range(self) -> float:
        """High - Low. Named to avoid shadowing built-in range()."""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Absolute candle body size."""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        """True if close > open."""
        return self.close > self.open

    @property
    def typical_price(self) -> float:
        """(H + L + C) / 3"""
        return (self.high + self.low + self.close) / 3.0

    @property
    def close_position(self) -> float:
        """
        Where close sits in the bar range.
        Returns value in [0, 1]. 0.5 for doji candles.
        0 = low, 1 = high.
        """
        r = self.bar_range
        return (self.close - self.low) / r if r > 1e-10 else 0.5

    @property
    def log_return(self) -> float:
        """Log return from open to close."""
        return math.log(self.close / (self.open + 1e-10))


@dataclass(frozen=True)
class OrderBookSnapshot:
    """
    Immutable snapshot of the order book at a point in time.

    Attributes:
        symbol: Trading pair
        timestamp: Unix timestamp
        bids: Sorted list of bid levels (highest first)
        asks: Sorted list of ask levels (lowest first)
    """
    symbol:    str
    timestamp: float
    bids:      Tuple[OrderBookLevel, ...]
    asks:      Tuple[OrderBookLevel, ...]

    @property
    def best_bid(self) -> Optional[float]:
        """Highest bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """Lowest ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        """Midpoint price."""
        b, a = self.best_bid, self.best_ask
        return (b + a) / 2.0 if (b is not None and a is not None) else None

    @property
    def spread(self) -> Optional[float]:
        """Absolute spread (ask - bid)."""
        b, a = self.best_bid, self.best_ask
        return a - b if (b is not None and a is not None) else None

    @property
    def spread_bps(self) -> Optional[float]:
        """Spread in basis points relative to mid."""
        mid = self.mid
        spd = self.spread
        return (spd / mid) * 10_000.0 if (mid and spd) else None

    def bid_depth(self, levels: int = 5) -> float:
        """Cumulative bid size for first N levels."""
        return sum(lv.size for lv in self.bids[:levels])

    def ask_depth(self, levels: int = 5) -> float:
        """Cumulative ask size for first N levels."""
        return sum(lv.size for lv in self.asks[:levels])

    @property
    def imbalance(self) -> float:
        """
        Book imbalance in [-1, 1].
        Positive = bid-heavy (buying pressure).
        Negative = ask-heavy (selling pressure).
        """
        bd = self.bid_depth()
        ad = self.ask_depth()
        total = bd + ad
        return (bd - ad) / total if total > 0 else 0.0


@dataclass(frozen=True)
class Indicators:
    """
    Technical indicators for a single bar.
    All fields Optional — None when not yet computed (warmup period).
    """
    # ── Trend indicators ──
    sma_20:       Optional[float] = None
    sma_50:       Optional[float] = None
    sma_200:      Optional[float] = None
    ema_12:       Optional[float] = None
    ema_26:       Optional[float] = None
    ema_50:       Optional[float] = None
    adx:          Optional[float] = None
    adx_plus_di:  Optional[float] = None
    adx_minus_di: Optional[float] = None

    # ── Momentum indicators ──
    rsi_14:       Optional[float] = None
    rsi_7:        Optional[float] = None
    macd_line:    Optional[float] = None
    macd_signal:  Optional[float] = None
    macd_hist:    Optional[float] = None
    stoch_k:      Optional[float] = None
    stoch_d:      Optional[float] = None
    williams_r:   Optional[float] = None
    cci:          Optional[float] = None
    mfi:          Optional[float] = None

    # ── Volatility indicators ──
    atr_14:       Optional[float] = None
    bb_upper:     Optional[float] = None
    bb_middle:    Optional[float] = None
    bb_lower:     Optional[float] = None
    bb_width:     Optional[float] = None
    bb_pct:       Optional[float] = None
    keltner_upper: Optional[float] = None
    keltner_lower: Optional[float] = None

    # ── Volume indicators ──
    obv:          Optional[float] = None
    vwap:         Optional[float] = None
    adl:          Optional[float] = None
    cmf:          Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Export non-None indicators as a dictionary."""
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        return result


@dataclass(frozen=True)
class SentimentSnapshot:
    """
    Market sentiment data for a point in time.
    All scores in [-1, 1] except confidence which is [0, 1].
    """
    symbol:       str
    timestamp:    float
    overall:      float    # [-1, 1] composite sentiment
    confidence:   float    # [0, 1] confidence in the signal
    news:         float = 0.0   # News sentiment
    social:       float = 0.0   # Social media sentiment
    options_flow: float = 0.0   # Options flow sentiment
    put_call:     float = 1.0   # Put/call ratio


@dataclass
class PositionState:
    """
    Mutable state for an open position.
    Updated each bar via mark_to_market.
    """
    symbol:         str
    side:           Side
    quantity:       float
    avg_entry:      float
    current_price:  float
    stop_loss:      Optional[float]
    take_profit:    Optional[float]
    hold_bars:      int = 0
    unrealised_pnl: float = 0.0
    max_drawdown:   float = 0.0
    peak_price:     float = 0.0

    def update(self, price: float) -> None:
        """
        Update position state with new mark price.
        Recalculates unrealised P&L and tracks drawdown.
        """
        self.current_price = price
        mult = 1.0 if self.side == Side.LONG else -1.0
        self.unrealised_pnl = (price - self.avg_entry) * self.quantity * mult

        if self.peak_price == 0.0:
            self.peak_price = price
        elif self.side == Side.LONG and price > self.peak_price:
            self.peak_price = price
        elif self.side == Side.SHORT and price < self.peak_price:
            self.peak_price = price

        if self.peak_price > 0:
            self.max_drawdown = max(
                0.0,
                (self.peak_price - price) / self.peak_price
            )

    @property
    def pnl_pct(self) -> float:
        """Unrealised P&L as percentage of entry notional."""
        denom = self.avg_entry * self.quantity
        return self.unrealised_pnl / denom if denom != 0 else 0.0

    @property
    def locked_capital(self) -> float:
        return abs(self.quantity * self.avg_entry)

    @property
    def notional(self) -> float:
        """Current position notional value."""
        return abs(self.quantity * self.current_price)

    @property
    def dist_to_sl(self) -> float:
        """Normalised distance to stop loss (0 = at SL, 1 = far)."""
        if self.stop_loss is None or self.avg_entry == 0:
            return 1.0
        return abs(self.current_price - self.stop_loss) / (self.avg_entry + 1e-10)

    @property
    def dist_to_tp(self) -> float:
        """Normalised distance to take profit (0 = at TP, 1 = far)."""
        if self.take_profit is None or self.avg_entry == 0:
            return 1.0
        return abs(self.take_profit - self.current_price) / (self.avg_entry + 1e-10)


@dataclass
class PortfolioSnapshot:
    """Complete portfolio state at a point in time."""
    total_capital:   float
    available_cash:  float
    total_exposure:  float
    exposure_pct:    float
    portfolio_heat:  float
    unrealised_pnl:  float
    realised_pnl:    float
    daily_pnl:       float
    position:        Optional[PositionState] = None

    @property
    def equity(self) -> float:
        """Total equity = capital + unrealised P&L."""
        return self.total_capital + self.unrealised_pnl


@dataclass(frozen=True)
class Action:
    """
    Decoded continuous action from the agent.

    Attributes:
        direction: [-1, 1] — negative=short, positive=long, near 0=flat
        sl_dist:   [0, 1]  — normalised stop distance (mapped to % of price)
        tp_mult:   [0, 1]  — normalised TP multiplier (mapped to multiple of SL)
    """
    direction: float
    sl_dist:   float = 0.5
    tp_mult:   float = 0.5

    @property
    def side(self) -> Side:
        """Decode direction into Side enum."""
        if self.direction > 0.05:
            return Side.LONG
        if self.direction < -0.05:
            return Side.SHORT
        return Side.FLAT

    @property
    def size(self) -> float:
        """Position size fraction [0, 1]."""
        return abs(self.direction)

    @property
    def sl_pct(self) -> float:
        """Convert sl_dist [0,1] → stop loss percentage [0.5%, 5.0%]."""
        return 0.005 + self.sl_dist * 0.045

    @property
    def tp_multiplier(self) -> float:
        """Convert tp_mult [0,1] → take profit multiplier [1.5×, 4.0×] of SL."""
        return 1.5 + self.tp_mult * 2.5

    @classmethod
    def flat(cls) -> "Action":
        """Convenience constructor for flat/close action."""
        return cls(direction=0.0)


@dataclass(frozen=True)
class Action:
    direction: float
    sl_dist:   float = 0.5
    tp_mult:   float = 0.5

    @property
    def side(self) -> Side:
        if self.direction > 0.02:
            return Side.LONG
        if self.direction < -0.02:
            return Side.SHORT
        return Side.FLAT

    @property
    def size(self) -> float:
        return abs(self.direction)

    @property
    def sl_pct(self) -> float:
        return 0.005 + self.sl_dist * 0.025

    @property
    def tp_multiplier(self) -> float:
        return 1.5 + self.tp_mult * 2.0


@dataclass
class StepResult:
    """
    Single step result from the environment.
    All observation fields are torch Tensors on the configured device.
    """
    obs_market_seq:    Tensor     # [lookback, N_MARKET_FEATURES]
    obs_portfolio_vec: Tensor     # [N_PORTFOLIO_FEATURES]
    obs_sentiment_vec: Tensor     # [N_SENTIMENT_FEATURES]
    obs_time_vec:      Tensor     # [N_TIME_FEATURES]
    reward:            float
    terminated:        bool
    truncated:         bool
    info:              Dict[str, Any]
    termination_cause: Optional[EpisodeTermination] = None

    @property
    def done(self) -> bool:
        """True if episode has ended for any reason."""
        return self.terminated or self.truncated


@dataclass
class EpisodeStats:
    """Summary statistics for a completed episode."""
    episode_id:     str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    total_reward:   float = 0.0
    total_steps:    int = 0
    total_trades:   int = 0
    winning_trades: int = 0
    realised_pnl:   float = 0.0
    max_drawdown:   float = 0.0
    peak_capital:   float = 0.0
    final_capital:  float = 0.0
    termination:    Optional[EpisodeTermination] = None

    @property
    def win_rate(self) -> float:
        """Fraction of winning trades."""
        return (
            self.winning_trades / self.total_trades
            if self.total_trades > 0 else 0.0
        )

    def record_trade(self, pnl: float) -> None:
        """Record a completed trade."""
        self.total_trades += 1
        self.realised_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1
