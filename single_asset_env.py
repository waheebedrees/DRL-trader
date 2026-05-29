
from __future__ import annotations

from abc import ABC, abstractmethod
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch import Tensor


from config import (
    EnvironmentConfig,
    EnvironmentNotResetError,
    EpisodeTerminatedError,
    InsufficientDataError,

)



from state import (

    EpisodeStats,
    Side,
    EpisodeTermination,
    ActionSpace,
    Action,
    Indicators,
    OrderBookSnapshot,
    SentimentSnapshot,
    StepResult,
    Bar,
    MarketStateBuilder,
)

from execution_simulator import ExecutionConfig, ExecutionSimulator
from reward_engine import RewardEngine


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


@dataclass
class StepContext:
    """Everything a reward scheme needs — decoupled from environment."""
    realised_pnl:       float
    realised_pnl_pct:   float
    total_pnl_pct:      float
    hit_sl:             bool
    hit_tp:             bool
    tp_level:           int
    closed:             bool
    opened:             bool
    drawdown:           float
    leverage:           float
    hold_bars:          int
    daily_trades:       int
    portfolio_value:    float
    position_side:      int
    price_return:       float
    unrealised_pnl_pct: float   # R-06: for hold bonus calculation


class SingleAssetEnv:
    DMAP: Dict[int, Tuple[float, float, float]] = {
        0: (0., 0.5, 0.5),    1: (0.33, 0.5, 0.5),  2: (1., 0.5, 0.5),
        3: (-0.33, 0.5, 0.5), 4: (-1., 0.5, 0.5),   5: (0., 0.5, 0.5),
    }

    def __init__(self, bars: List[Bar], cfg: EnvironmentConfig, device: torch.device,
                 indicators: Optional[List[Optional[Indicators]]] = None,
                 orderbooks: Optional[List[Optional[OrderBookSnapshot]]] = None,
                 sentiment:  Optional[List[Optional[SentimentSnapshot]]] = None) -> None:
        if not bars:
            raise ValueError("bars list is empty")
        self.cfg = cfg
        self.device = device
        self.bars = bars
        self.inds = indicators or [None] * len(bars)
        self.obs_ = orderbooks or [None] * len(bars)
        self.sent = sentiment or [None] * len(bars)
        self._sb = MarketStateBuilder(
            cfg.lookback_window, device, cfg.normalise_obs, cfg.clip_obs)
        self._re = RewardEngine(cfg.reward, device)
        self._ex = ExecutionSimulator(cfg.initial_capital,
                                      ExecutionConfig(cfg.commission_rate, cfg.slippage_rate,
                                                      cfg.funding_rate_per_bar, cfg.max_leverage))
        self._start_idx: int = cfg.warmup_bars
        self._idx = 0
        self._dt = 0
        self._pk = cfg.initial_capital
        self._done = False
        self._rc = False
        self._term: Optional[EpisodeTermination] = None
        self._stats = EpisodeStats()
        self._max_start = max(
            cfg.warmup_bars,
            len(bars) - cfg.episode_length - cfg.warmup_bars,
        )

    def reset(self, seed: Optional[int] = None) -> StepResult:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        if self.cfg.random_start and self._max_start > self.cfg.warmup_bars:
            self._start_idx = random.randint(
                self.cfg.warmup_bars, self._max_start)
        else:
            self._start_idx = self.cfg.warmup_bars
        self._idx = self._start_idx
        self._dt = 0
        self._pk = self.cfg.initial_capital
        self._done = False
        self._rc = True
        self._term = None
        self._stats = EpisodeStats(peak_capital=self.cfg.initial_capital,
                                   final_capital=self.cfg.initial_capital)
        self._sb.reset()
        self._re.reset()
        self._ex.reset(self.cfg.initial_capital)
        warmup_from = max(0, self._start_idx - self.cfg.warmup_bars)
        for i in range(warmup_from, self._start_idx):
            self._sb.update(self.bars[i])
        return StepResult(*self._obs(), 0., False, False, {"reset": True})

    def step(self, action: np.ndarray) -> StepResult:
        if not self._rc:
            raise EnvironmentNotResetError()
        if self._done:
            raise EpisodeTerminatedError(
                self._term.value if self._term else "?")
        if self._idx >= len(self.bars):
            self._done = True
            self._term = EpisodeTermination.DATA_EXHAUSTED
            return StepResult(*self._zero(), 0., False, True, {}, self._term)
        bar = self.bars[self._idx]
        self._sb.update(bar)
        ctx = self._exec(bar, self._parse(action))
        rew, comps = self._re.step(ctx)
        rew = float(np.clip(rew, -100, 100))
        self._idx += 1
        self._stats.total_steps += 1
        self._stats.total_reward += rew
        term, cause = self._check_term()
        trunc = (self._idx >= self._start_idx + self.cfg.episode_length
                 or self._idx >= len(self.bars))
        self._term = cause if term else (
            EpisodeTermination.DATA_EXHAUSTED if trunc else None)
        self._done = term or trunc
        if self._done:
            s = self._ex.snap()
            self._stats.final_capital = s.total_capital
            self._stats.max_drawdown = self._dd()
            self._stats.termination = self._term
        info = {"capital": self._ex.capital, "drawdown": self._dd(), "step": self._idx,
                "daily_trades": self._dt, "reward_components": comps,
                "position": self._ex.position is not None,
                "termination": self._term.value if self._term else None}
        if self._done:
            info["episode_stats"] = self._stats
        return StepResult(*self._obs(), rew, term, trunc, info, self._term)

    def _parse(self, a: np.ndarray) -> Action:
        if self.cfg.action_space == ActionSpace.DISCRETE:
            idx = max(0, min(int(np.asarray(a).flat[0]), len(self.DMAP) - 1))
            d, s, t = self.DMAP[idx]
            return Action(d, s, t)
        a = np.clip(np.asarray(a, np.float32).flatten(), -1, 1)
        return Action(float(a[0]) if len(a) > 0 else 0.,
                      float((a[1] + 1) / 2) if len(a) > 1 else .5,
                      float((a[2] + 1) / 2) if len(a) > 2 else .5)



    def _exec(self, bar: Bar, action: Action) -> StepContext:
        price = bar.close
        prev_cap = self._ex.capital
        hit_sl = hit_tp = opened = closed = False
        tp_lvl = 0
        realised_pnl = 0.0

        _, hit_sl, hit_tp = self._ex.mtm(price)

        if hit_sl:
            realised_pnl, _ = self._ex.close(price, "stop_loss")
            self._stats.record_trade(realised_pnl)
            closed = True
        elif hit_tp:
            realised_pnl, _ = self._ex.close(price, "take_profit")
            self._stats.record_trade(realised_pnl)
            closed = True
            tp_lvl = 1

        if not closed:
            cs = self._side()
            ts = action.side
            if ts == Side.FLAT and self._ex.position:
                realised_pnl, _ = self._ex.close(price, "agent_close")
                self._stats.record_trade(realised_pnl)
                closed = True
            elif ts != Side.FLAT and ts != cs:
                if self._ex.position:
                    realised_pnl, _ = self._ex.close(price, "flip")
                    self._stats.record_trade(realised_pnl)
                    closed = True
                _, fill = self._ex.open(
                    price, ts, action.size, action.sl_pct, action.tp_multiplier)
                if not fill.get("skipped"):
                    self._dt += 1
                    self._stats.total_trades += 1
                    opened = True

        snap = self._ex.snap()
        self._pk = max(self._pk, snap.total_capital)
        total_net = snap.total_capital - prev_cap

        realised_pct = float(
            np.clip(realised_pnl / (prev_cap + 1e-10), -1, 1)) if closed else 0.0
        total_pct = float(np.clip(total_net / (prev_cap + 1e-10), -1, 1))
        # R-06: unrealised pnl pct for hold bonus
        unrealised_pct = float(
            np.clip(snap.unrealised_pnl / (snap.total_capital + 1e-10), -1, 1))

        pos = self._ex.position
        ps = 1 if (pos and pos.side == Side.LONG) else -1 if pos else 0
        pb = self.bars[self._idx - 1].close if self._idx > 0 else bar.close
        br = float(np.clip((bar.close - pb) / (pb + 1e-10), -.5, .5))

        return StepContext(
            realised_pnl=realised_pnl,
            realised_pnl_pct=realised_pct,
            total_pnl_pct=total_pct,
            hit_sl=hit_sl, hit_tp=hit_tp, tp_level=tp_lvl,
            closed=closed, opened=opened,
            drawdown=self._dd(), leverage=self._lev(),
            hold_bars=pos.hold_bars if pos else 0,
            daily_trades=self._dt, portfolio_value=snap.total_capital,
            position_side=ps, price_return=br,
            unrealised_pnl_pct=unrealised_pct,
        )

    def _obs(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        idx = min(self._idx, len(self.bars) - 1)
        try:
            return self._sb.build(self.bars[idx], self.inds[idx], self.obs_[idx],
                                  self._ex.snap(), self.sent[idx])
        except (InsufficientDataError, IndexError):
            return self._zero()

    def _zero(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        lw = self.cfg.lookback_window
        return (torch.zeros(lw, N_MARKET_FEATURES,    device=self.device),
                torch.zeros(N_PORTFOLIO_FEATURES,      device=self.device),
                torch.zeros(N_SENTIMENT_FEATURES,      device=self.device),
                torch.zeros(N_TIME_FEATURES,           device=self.device))

    def _dd(self) -> float:
        pk = self._pk
        return float(np.clip((pk - self._ex.capital) / pk, 0, 1)) if pk > 0 else 0.

    def _lev(self) -> float:
        pos = self._ex.position
        cap = self._ex.capital
        return min(pos.notional / cap, 10.) if pos and cap > 0 else 1.

    def _side(self) -> Side:
        return self._ex.position.side if self._ex.position else Side.FLAT

    def _check_term(self) -> Tuple[bool, Optional[EpisodeTermination]]:
        if self._dd() >= self.cfg.max_drawdown_terminate:
            return True, EpisodeTermination.MAX_DRAWDOWN
        if self._ex.capital < self.cfg.initial_capital * self.cfg.min_capital_pct:
            return True, EpisodeTermination.MIN_CAPITAL
        if self._idx >= self._start_idx + self.cfg.episode_length:
            return True, EpisodeTermination.MAX_STEPS
        return False, None

    @property
    def is_done(self) -> bool: return self._done

    @property
    def action_dim(self) -> int:
        return self.cfg.n_discrete_actions if self.cfg.action_space == ActionSpace.DISCRETE else 3

    @property
    def episode_stats(self) -> EpisodeStats: return self._stats
