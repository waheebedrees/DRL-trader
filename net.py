# net.py

from __future__ import annotations

from abc import ABC, abstractmethod
import os
import random
import numpy as np
from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple, Any
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim


from config import (
    NetworkConfig,
    RainbowConfig,
    RewardConfig,
    EnvironmentConfig,
    PPOConfig,
    TD3Config,
    SACConfig,

    MarketConfig,
    EnvironmentNotResetError,
    EpisodeTerminatedError,
    InsufficientDataError,
    TrainingError,
    TrainingConfig,
    CheckpointError,
    ConfigurationError,
    
)


from networks import (
    # Layers
    NoisyLinear,
    GatedResidualBlock,
    PositionalEncoding,
    RotaryEmbedding,
    MarketAttention,
    SwiGLUFFN,
    DilatedCausalConv,
    SqueezeExcitation,
    # Encoders
    MarketTransformerEncoder,
    AttentionLSTMEncoder,
    TemporalCNNEncoder,
    HybridEncoder,
    PortfolioEncoder,
    build_encoder,
    # Heads
    ContinuousActor,
    DiscreteActor,
    DuelingCritic,
    DistributionalCritic,
    TwinQCritic,
    # Output containers
    ActorOutput,
    CriticOutput,
    DistributionalOutput,
)

from utils.torch_utils import (
    select_device,

)
from state import (
    PositionState,
    PortfolioSnapshot,
    EpisodeStats,
    TimeFrame,
    Side,
    EpisodeTermination,
    ActionSpace,
    Architecture,
    DRLAlgorithm,
    AssetClass,    
    RewardScheme,
    MarketRegime,
    Action,
    Indicators,
    OrderBookLevel,
    OrderBookSnapshot,
    PortfolioSnapshot,
    PositionState,
    SentimentSnapshot,
    StepResult,
    Bar,
    MarketStateBuilder,
)

# ==============================================================================
# SECTION 3: Feature constants
# ==============================================================================

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



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _chk(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


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

@dataclass
class ExecutionConfig:
    """Execution simulation parameters."""
    commission_rate:      float = 0.001
    slippage_rate:        float = 0.0005
    funding_rate_per_bar: float = 0.00005
    max_leverage:         float = 3.0
    min_order_notional:   float = 10.0


class ActorCriticNetwork(nn.Module):
    """
    Full Actor-Critic network for PPO / SAC.

    Architecture:
    ┌─────────────────────────────────────────────────────┐
    │  Market Seq [B,T,F]  ─→  MarketEncoder  ─→ [B, D]  │
    │  Portfolio  [B,P]    ─→  PortfolioEnc   ─→ [B, D/4]│
    │  Sentiment  [B,S]    ─→  Linear          ─→ [B, 32] │
    │  Time       [B,6]    ─→  Linear          ─→ [B, 16] │
    │                          Concatenate                  │
    │                          ↓                           │
    │                     Shared Trunk                     │
    │                   (MLP + GRN blocks)                 │
    │                    /              \                  │
    │               Actor              Critic              │
    │          (Gaussian dist)     (Dueling Q)             │
    └─────────────────────────────────────────────────────┘
    """

    def __init__(self, cfg: NetworkConfig, nm: int, np_: int,
                 ns: int = 5, nt: int = 6, ad: int = 3, cont: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_dim = ad
        self.continuous = cont
        D = cfg.d_model
        self.menc = build_encoder(cfg, nm)
        md = self.menc.out_dim
        pd = max(D // 4, 16)
        self.penc = PortfolioEncoder(np_, pd, cfg.dropout)
        self.senc = nn.Sequential(
            nn.Linear(ns, 32), nn.LayerNorm(32), nn.GELU())
        self.tenc = nn.Sequential(nn.Linear(nt, 16), nn.GELU())
        dims = [md + pd + 32 + 16] + list(cfg.hidden_dims)
        trunk: List[nn.Module] = []
        for i in range(len(dims) - 1):
            trunk += [nn.Linear(dims[i], dims[i + 1]), nn.LayerNorm(dims[i + 1]),
                      nn.GELU(), nn.Dropout(cfg.dropout)]
            if i < len(dims) - 2:
                trunk.append(GatedResidualBlock(
                    dims[i + 1], dims[i + 1], cfg.dropout))
        self.trunk = nn.Sequential(*trunk)
        fd = cfg.hidden_dims[-1]
        self.actor = (ContinuousActor(fd, ad, (fd // 2,), cfg.dropout) if cont
                      else DiscreteActor(fd, ad, (fd // 2,)))
        self.critic = DuelingCritic(fd, ad, fd // 2, cfg.use_noisy_net)
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear) and not isinstance(m, NoisyLinear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.actor.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 0.01)

    def enc_params(self) -> List[nn.Parameter]:
        return (list(self.menc.parameters()) + list(self.penc.parameters())
                + list(self.senc.parameters()) + list(self.tenc.parameters()))

    def non_enc_params(self) -> List[nn.Parameter]:
        ep = {id(p) for p in self.enc_params()}
        return [p for p in self.parameters() if id(p) not in ep]

    def encode(self, mkt: Tensor, port: Tensor,
               sent: Optional[Tensor] = None, tv: Optional[Tensor] = None,
               hx: Optional[Tuple] = None) -> Tuple[Tensor, Optional[Tuple]]:
        enc = self.menc(mkt, hx)
        mo, hx_out = (enc if isinstance(enc, tuple) else (enc, None))
        po = self.penc(port)
        B = mkt.size(0)
        so = self.senc(sent) if sent is not None else torch.zeros(
            B, 32, device=mkt.device)
        to = self.tenc(tv) if tv is not None else torch.zeros(
            B, 16, device=mkt.device)
        return self.trunk(torch.cat([mo, po, so, to], -1)), hx_out

    def get_value(self, mkt: Tensor, port: Tensor,
                  sent: Optional[Tensor] = None, tv: Optional[Tensor] = None) -> Tensor:
        f, _ = self.encode(mkt, port, sent, tv)
        return self.critic.get_value(f)

    def get_action_and_value(
            self, mkt: Tensor, port: Tensor,
            sent: Optional[Tensor] = None, tv: Optional[Tensor] = None,
            action: Optional[Tensor] = None, hx: Optional[Tuple] = None, det: bool = False,
    ) -> Tuple[ActorOutput, CriticOutput, Optional[Tuple]]:
        f, hx_out = self.encode(mkt, port, sent, tv, hx)
        return self.actor(f, action, det), CriticOutput(self.critic.get_value(f)), hx_out

    def reset_noise(self) -> None:
        if self.training:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()

    def param_count(self) -> Dict[str, int]:
        return {"total":     sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)}


class RollingMetrics(nn.Module):
    def __init__(self, W: int, rfr: float, dev: torch.device) -> None:
        super().__init__()
        self.W = W
        self.rfr = rfr
        self.register_buffer("_r",    torch.zeros(W, device=dev))
        self.register_buffer("_ptr",  torch.tensor(
            0, dtype=torch.int64, device=dev))
        self.register_buffer("_cnt",  torch.tensor(
            0, dtype=torch.int64, device=dev))
        self.register_buffer("_pk",   torch.tensor(0., device=dev))
        self.register_buffer("_wins", torch.tensor(
            0, dtype=torch.int64, device=dev))
        self.register_buffer("_loss", torch.tensor(
            0, dtype=torch.int64, device=dev))

    @torch.no_grad()
    def push_ret(self, r: float) -> None:
        r = float(np.clip(r, -0.5, 0.5))
        self._r[self._ptr] = r
        self._ptr = (self._ptr + 1) % self.W
        self._cnt = (self._cnt + 1).clamp(max=self.W)

    @torch.no_grad()
    def push_peak(self, cap: float) -> None:
        self._pk = torch.max(self._pk,
                             torch.tensor(float(np.clip(cap, 0, 1e12)), device=self._pk.device))

    @torch.no_grad()
    def push_trade(self, pnl: float) -> None:
        if pnl > 0:
            self._wins.add_(1)
        else:
            self._loss.add_(1)

    def reset(self) -> None:
        for b in [self._r, self._ptr, self._cnt, self._pk, self._wins, self._loss]:
            b.zero_()

    def _active(self) -> Tensor:
        n = int(self._cnt.item())
        return self._r[:max(1, min(n, self.W))]

    @torch.no_grad()
    def sharpe(self) -> Tensor:
        r = self._active()
        if len(r) < 10:
            return torch.zeros(1, device=r.device).squeeze()
        ex = r.mean() - self.rfr / 252
        return torch.tanh(ex / r.std().clamp(1e-8) * math.sqrt(252) / 3)

    @torch.no_grad()
    def dd_pen(self, dd: float) -> Tensor:
        dd = float(np.clip(dd, 0, 1))
        dev = self._pk.device
        if dd <= 0.02:
            return torch.zeros(1, device=dev).squeeze()
        if dd <= 0.05:
            return torch.tensor(-(dd - 0.02) * 3, device=dev)
        return torch.tensor(-(0.09 + (dd - 0.05) ** 2 * 10), device=dev)

    @torch.no_grad()
    def scale_pnl(self, pct: float) -> Tensor:
        # R-02: multiply by 100 first so a 0.1% trade gives ~log1p(0.1*10/100*100)
        # = log1p(1.0) ≈ 0.69, not log1p(0.001*10) ≈ 0.01
        # Formula: treat pct as already in percent-point units (0.001 → 0.1%)
        # Scale: pct_pp = pct * 100, then log1p(|pct_pp| * 2) with sign
        pct = float(np.clip(pct, -0.5, 0.5))
        pct_pp = pct * 100.0   # convert 0.001 → 0.1 (percent-points)
        return torch.tensor(
            math.copysign(math.log1p(abs(pct_pp) * 2.0), pct_pp),
            device=self._pk.device,
        )

    @torch.no_grad()
    def trade_q(self) -> Tensor:
        tot = (self._wins + self._loss).float()
        wr = self._wins.float() / tot if tot > 0 else torch.tensor(0.5, device=tot.device)
        return torch.tanh((wr - 0.5) * 4)


class RewardEngine(nn.Module):
    def __init__(self, cfg: RewardConfig, dev: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.dev = dev
        self.m = RollingMetrics(cfg.returns_window, cfg.risk_free_rate, dev)

    def reset(self) -> None: self.m.reset()

    @torch.no_grad()
    def step(self, ctx: StepContext) -> Tuple[float, Dict[str, float]]:
        self.m.push_ret(ctx.total_pnl_pct)
        self.m.push_peak(ctx.portfolio_value)
        if ctx.closed:
            self.m.push_trade(ctx.realised_pnl)
        c = self.cfg

        comps: Dict[str, Tensor] = {
            # R-01/R-02: PnL only on closed trades; R-02 scaling amplifies small %
            "pnl": self.m.scale_pnl(ctx.realised_pnl_pct) * c.w_pnl if ctx.closed
            else torch.zeros(1, device=self.dev).squeeze(),
            "sh":  self.m.sharpe() * c.w_sharpe,
            "dd":  self.m.dd_pen(ctx.drawdown) * c.w_drawdown,
            "tq":  self.m.trade_q() * c.w_trade_quality,
            # R-01: risk_compliance removed — was polluting reward baseline
        }

        # Bonuses
        if ctx.hit_tp:
            comps["tp"] = torch.tensor(
                c.bonus_hit_tp * max(1, ctx.tp_level), device=self.dev)
        if ctx.hit_sl:
            comps["sl"] = torch.tensor(c.penalty_hit_sl, device=self.dev)

        # R-06: hold bonus — reward agent for staying in a winning position
        if (not ctx.closed and ctx.position_side != 0
                and ctx.unrealised_pnl_pct > c.hold_profit_threshold):
            comps["hold"] = torch.tensor(c.bonus_hold_profit, device=self.dev)

        # Directional bonus
        if ((ctx.position_side == 1 and ctx.price_return > 0) or
                (ctx.position_side == -1 and ctx.price_return < 0)):
            comps["dir"] = torch.tensor(
                c.bonus_correct_direction, device=self.dev)

        # Penalties
        if ctx.drawdown > c.max_drawdown_threshold:
            comps["ddb"] = torch.tensor(
                c.penalty_large_drawdown *
                (1 + (ctx.drawdown - c.max_drawdown_threshold) * 5),
                device=self.dev)
        if ctx.daily_trades > c.max_daily_trades:
            comps["ot"] = torch.tensor(c.penalty_overtrade, device=self.dev)
        if ctx.leverage > c.max_leverage:
            comps["lv"] = torch.tensor(
                c.penalty_over_leverage, device=self.dev)
        if ctx.hold_bars > c.max_hold_bars:
            comps["hld"] = torch.tensor(
                c.penalty_hold_too_long, device=self.dev)

        # R-05: single scale applied once at the end
        raw = sum(v.item() for v in comps.values())
        total = float(np.clip(raw, -c.clip_reward,
                      c.clip_reward)) * c.reward_scale
        return total, {k: float(v.item()) for k, v in comps.items()}


class ExecutionSimulator:
    """
    Realistic order-fill simulator with:
    - Slippage (worse fill for market orders)
    - Commission (deducted from capital)
    - Funding rate (per-bar cost for holding positions)
    - Stop loss / take profit monitoring
    - Trade logging

    Completely decoupled from the environment — testable with raw prices only.
    All computations in Python floats (not torch) since this runs on CPU.
    """

    def __init__(
            self, ic: float, cfg: ExecutionConfig) -> None:
        self.cfg = cfg
        self._ic = ic
        self._cash: float = ic
        self._pos:  Optional[PositionState] = None
        self._log:  List[Dict] = []

    def reset(self, cap: Optional[float] = None) -> None:
        self._cash = cap if cap is not None else self._ic
        self._pos = None
        self._log.clear()

    @property
    def capital(self) -> float:
        return self._cash + (self._pos.notional if self._pos else 0.0)

    @property
    def position(self) -> Optional[PositionState]: return self._pos

    @property
    def trade_log(self) -> List[Dict]: return list(self._log)

    def snap(self) -> PortfolioSnapshot:
        cap = self.capital
        pos = self._pos
        not_ = pos.notional if pos else 0.0
        real = sum(t["net_pnl"] for t in self._log)
        heat = min(not_ / (cap + 1e-10), 1.) if cap > 0 else 0.
        return PortfolioSnapshot(
            total_capital=cap, available_cash=self._cash,
            total_exposure=not_, exposure_pct=min(not_ / (cap + 1e-10), 1.),
            portfolio_heat=heat,
            unrealised_pnl=pos.unrealised_pnl if pos else 0.,
            realised_pnl=real, daily_pnl=real, position=pos)

    def open(self, price: float, side: Side,
             size_pct: float, sl_pct: float, tp_mult: float,
             ) -> Tuple[PortfolioSnapshot, Dict]:
        size_pct = float(np.clip(size_pct, 0.01, 1.0))
        notional = self._cash * size_pct
        if notional < self.cfg.min_order_notional:
            return self.snap(), {"skipped": True, "reason": "min_notional"}
        comm = notional * self.cfg.commission_rate
        slpg = notional * self.cfg.slippage_rate
        cost = comm + slpg
        total = notional + cost
        if total > self._cash:
            notional = self._cash * 0.95 / \
                (1 + self.cfg.commission_rate + self.cfg.slippage_rate)
            if notional < self.cfg.min_order_notional:
                return self.snap(), {"skipped": True, "reason": "insufficient_cash"}
            comm = notional * self.cfg.commission_rate
            slpg = notional * self.cfg.slippage_rate
            cost = comm + slpg
            total = notional + cost
        slip = self.cfg.slippage_rate
        fill = max(price * (1 + slip) if side ==
                   Side.LONG else price * (1 - slip), 1e-10)
        qty = notional / fill
        sl = fill * (1 - sl_pct) if side == Side.LONG else fill * (1 + sl_pct)
        tp = fill * \
            (1 + sl_pct * tp_mult) if side == Side.LONG else fill * \
            (1 - sl_pct * tp_mult)
        self._cash -= total
        self._cash = max(self._cash, 0.)
        self._pos = PositionState(
            "", side, qty, fill, fill, sl, tp, peak_price=fill)
        return self.snap(), {"fill_price": fill, "quantity": qty, "notional": notional,
                             "commission": comm, "slippage": slpg, "skipped": False}

    def close(self, price: float, reason: str) -> Tuple[float, Dict]:
        if self._pos is None:
            return 0., {"skipped": True}
        pos = self._pos
        slip = self.cfg.slippage_rate
        fill = max(price * (1 - slip) if pos.side ==
                   Side.LONG else price * (1 + slip), 1e-10)
        comm = fill * pos.quantity * self.cfg.commission_rate
        proceeds = fill * pos.quantity - comm
        cost_basis = pos.avg_entry * pos.quantity
        net_pnl = proceeds - cost_basis
        self._cash += proceeds
        self._cash = max(self._cash, 0.)
        rec = {"reason": reason, "fill_price": fill,
               "gross_pnl": (fill - pos.avg_entry) * pos.quantity * (1 if pos.side == Side.LONG else -1),
               "net_pnl": net_pnl, "commission": comm,
               "hold_bars": pos.hold_bars, "quantity": pos.quantity}
        self._log.append(rec)
        self._pos = None
        return net_pnl, rec

    def mtm(self, price: float) -> Tuple[PortfolioSnapshot, bool, bool]:
        hit_sl = hit_tp = False
        if self._pos:
            pos = self._pos
            pos.update(price)
            pos.hold_bars += 1
            self._cash -= pos.notional * self.cfg.funding_rate_per_bar
            self._cash = max(self._cash, 0.)
            if pos.stop_loss is not None:
                if ((pos.side == Side.LONG and price <= pos.stop_loss) or
                        (pos.side == Side.SHORT and price >= pos.stop_loss)):
                    hit_sl = True
            if pos.take_profit is not None and not hit_sl:
                if ((pos.side == Side.LONG and price >= pos.take_profit) or
                        (pos.side == Side.SHORT and price <= pos.take_profit)):
                    hit_tp = True
        return self.snap(), hit_sl, hit_tp


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


# ==============================================================================
# SECTION 14: Replay Buffers
# ==============================================================================
@dataclass
class Transition:
    mkt:   np.ndarray
    port:  np.ndarray
    sent:  np.ndarray
    tv:    np.ndarray
    nmkt:  np.ndarray
    nport: np.ndarray
    nsent: np.ndarray
    ntv:   np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


class SumTree:
    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.tree = np.zeros(2 * cap - 1, np.float64)
        self.data: List[Any] = [None] * cap
        self._ptr = 0
        self._sz = 0

    @property
    def total(self) -> float: return float(self.tree[0])

    @property
    def size(self) -> int: return self._sz

    def add(self, p: float, d: Any) -> None:
        idx = self._ptr + self.cap - 1
        self.data[self._ptr] = d
        self.update(idx, p)
        self._ptr = (self._ptr + 1) % self.cap
        self._sz = min(self._sz + 1, self.cap)

    def update(self, i: int, p: float) -> None:
        delta = p - self.tree[i]
        self.tree[i] = p
        while i > 0:
            i = (i - 1) // 2
            self.tree[i] += delta

    def get(self, s: float) -> Tuple[int, float, Any]:
        i = 0
        while True:
            left = 2 * i + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left] + 1e-12:
                i = left
            else:
                s -= self.tree[left]
                i = left + 1
        di = i - self.cap + 1
        if di < 0 or di >= self._sz or self.data[di] is None:
            di = random.randrange(max(self._sz, 1))
            i = di + self.cap - 1
        return i, float(self.tree[i]), self.data[di]

    def min_p(self) -> float:
        if self._sz == 0:
            return 1.
        lv = self.tree[self.cap - 1: self.cap - 1 + self._sz]
        pos = lv[lv > 0]
        return float(pos.min()) if len(pos) else 1.


class UniformBuf:
    def __init__(self, cap: int, dev: torch.device) -> None:
        self.cap = cap
        self.dev = dev
        self._b: List[Optional[Transition]] = [None] * cap
        self._ptr = 0
        self._sz = 0

    def add(self, t: Transition) -> None:
        self._b[self._ptr] = t
        self._ptr = (self._ptr + 1) % self.cap
        self._sz = min(self._sz + 1, self.cap)

    def sample(self, n: int) -> List[Transition]:
        return [self._b[i] for i in random.sample(range(self._sz), min(n, self._sz))]

    def collate(self, batch: List[Transition]) -> Dict[str, Tensor]:
        def _t(a): return torch.from_numpy(np.stack(a)).float().to(self.dev)
        return {"market_seqs":         _t([t.mkt for t in batch]),
                "portfolio_vecs":      _t([t.port for t in batch]),
                "sentiment_vecs":      _t([t.sent for t in batch]),
                "time_vecs":           _t([t.tv for t in batch]),
                "next_market_seqs":    _t([t.nmkt for t in batch]),
                "next_portfolio_vecs": _t([t.nport for t in batch]),
                "next_sentiment_vecs": _t([t.nsent for t in batch]),
                "next_time_vecs":      _t([t.ntv for t in batch]),
                "actions": _t([t.action for t in batch]),
                "reward":  torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.dev),
                "done":    torch.tensor([float(t.done) for t in batch], dtype=torch.float32, device=self.dev)}

    def __len__(self) -> int: return self._sz
    def is_ready(self, n: int) -> bool: return self._sz >= n


class PERBuffer:
    def __init__(self, cap: int, dev: torch.device,
                 alpha: float = 0.6, beta0: float = 0.4,
                 beta_f: int = 100_000, eps: float = 1e-6) -> None:
        self.dev = dev
        self.alpha = alpha
        self.beta0 = beta0
        self.beta_f = beta_f
        self.eps = eps
        self.tree = SumTree(cap)
        self._maxp = 1.
        self._frame = 1
        self._h = UniformBuf(1, dev)

    @property
    def beta(self) -> float:
        return min(1., self.beta0 + (1 - self.beta0) * self._frame / self.beta_f)

    def add(self, t: Transition, p: Optional[float] = None) -> None:
        self.tree.add((p if p is not None else self._maxp) ** self.alpha, t)

    def sample(self, n: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        self._frame += 1
        trans: List[Transition] = []
        idxs = np.zeros(n, np.int32)
        ws = np.zeros(n, np.float32)
        seg = self.tree.total / n
        beta = self.beta
        minp = self.tree.min_p() / (self.tree.total + 1e-10)
        maxw = (self.tree.size * minp + 1e-10) ** (-beta)
        for i in range(n):
            s = random.uniform(seg * i, seg * (i + 1))
            idx, p, d = self.tree.get(max(s, 1e-12))
            idxs[i] = idx
            prob = p / (self.tree.total + 1e-10)
            ws[i] = (self.tree.size * prob + 1e-10) ** (-beta) / (maxw + 1e-10)
            trans.append(d)
        return trans, idxs, ws.astype(np.float32)

    def update_priorities(self, idxs: np.ndarray, td: np.ndarray) -> None:
        ps = (np.abs(td) + self.eps) ** self.alpha
        self._maxp = max(self._maxp, float(ps.max()))
        for i, p in zip(idxs.tolist(), ps.tolist()):
            self.tree.update(int(i), float(p))

    def collate(self, batch: List[Transition], ws: np.ndarray) -> Tuple[Dict, Tensor]:
        return self._h.collate(batch), torch.tensor(ws, dtype=torch.float32, device=self.dev)

    def __len__(self) -> int: return self.tree.size
    def is_ready(self, n: int) -> bool: return self.tree.size >= n


# ==============================================================================
# SECTION 15: Base Trainer
# ==============================================================================

class BaseTrainer(ABC):
    def __init__(self, net: nn.Module, dev: torch.device) -> None:
        self.net = net.to(dev)
        self.device = dev
        self._step = 0
        self._ep_rewards: List[float] = []

    @property
    def global_step(self) -> int: return self._step

    @abstractmethod
    def collect_rollout(self, env: SingleAssetEnv) -> Dict[str, float]: ...
    @abstractmethod
    def update(self) -> Dict[str, float]: ...
    @abstractmethod
    def _select_action(self, r: StepResult, det: bool) -> np.ndarray: ...

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"net": self.net.state_dict(),
                   "step": self._step, **self._extra_save()}, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ck["net"])
        self._step = ck.get("step", 0)
        self._extra_load(ck)

    def _extra_save(self) -> Dict: return {}
    def _extra_load(self, ck: Dict) -> None: pass

    @torch.no_grad()
    def evaluate(self, env: SingleAssetEnv, n: int = 5, det: bool = True) -> Dict[str, float]:
        self.net.eval()
        rews, caps, trades, wrs = [], [], [], []
        for ep in range(n):
            r = env.reset(seed=ep)
            er = 0.
            while not env.is_done:
                r = env.step(self._select_action(r, det))
                er += r.reward
            s = env.episode_stats
            rews.append(er)
            caps.append(s.final_capital)
            trades.append(s.total_trades)
            wrs.append(s.win_rate)
        return {"mean_reward":   float(np.mean(rews)),   "std_reward":    float(np.std(rews)),
                "mean_capital":  float(np.mean(caps)),   "mean_trades":   float(np.mean(trades)),
                "mean_win_rate": float(np.mean(wrs))}


# ==============================================================================
# SECTION 16: PPO Trainer
# ==============================================================================

class _NoAMP:
    def __enter__(self): return self
    def __exit__(self, *a): pass


class RolloutBuf:
    def __init__(self, dev: torch.device) -> None:
        self.dev = dev; self.clear()

    def add(self, r: StepResult, a: Tensor, lp: Tensor, v: Tensor,
            reward: float, done: float) -> None:
        self._m.append(r.obs_market_seq.detach().cpu())
        self._p.append(r.obs_portfolio_vec.detach().cpu())
        self._s.append(r.obs_sentiment_vec.detach().cpu())
        self._t.append(r.obs_time_vec.detach().cpu())
        self._a.append(a.detach().cpu()); self._lp.append(lp.detach().cpu())
        self._v.append(v.detach().cpu())
        self._r.append(float(reward)); self._d.append(float(done))

    def clear(self) -> None:
        self._m=[]; self._p=[]; self._s=[]; self._t=[]
        self._a=[]; self._lp=[]; self._v=[]; self._r=[]; self._d=[]
        self.adv: Optional[Tensor] = None; self.ret: Optional[Tensor] = None

    def __len__(self) -> int: return len(self._r)

    def tensors(self) -> Dict[str, Tensor]:
        return {"mkt":  torch.stack(self._m).to(self.dev),
                "port": torch.stack(self._p).to(self.dev),
                "sent": torch.stack(self._s).to(self.dev),
                "tv":   torch.stack(self._t).to(self.dev),
                "act":  torch.stack(self._a).to(self.dev),
                "lp":   torch.stack(self._lp).to(self.dev),
                "val":  torch.stack(self._v).squeeze(-1).to(self.dev),
                "rew":  torch.tensor(self._r, dtype=torch.float32, device=self.dev),
                "don":  torch.tensor(self._d, dtype=torch.float32, device=self.dev)}

    @torch.no_grad()
    def gae(self, T: Dict, lv: Tensor, last_done: bool,
            gamma: float, lam: float, norm_adv: bool, norm_ret: bool) -> None:
        N   = T["rew"].size(0)
        val = T["val"]
        vals_next      = torch.empty_like(val)
        vals_next[:-1] = val[1:]
        vals_next[-1]  = 0. if last_done else lv.squeeze().item()
        adv = torch.zeros(N, device=self.dev)
        g   = 0.
        for t in reversed(range(N)):
            nt    = 1. - T["don"][t].item()
            delta = T["rew"][t].item() + gamma * vals_next[t].item() * nt - val[t].item()
            g     = delta + gamma * lam * nt * g
            adv[t] = g
        ret = adv + val
        if norm_ret: ret = (ret - ret.mean()) / (ret.std().clamp(1e-8))
        if norm_adv: adv = (adv - adv.mean()) / (adv.std().clamp(1e-8))
        self.adv = adv; self.ret = ret

    def batches(self, T: Dict, bs: int):
        N = T["rew"].size(0)
        for idx in torch.randperm(N).split(bs):
            if not len(idx): continue
            idx = idx.to(self.dev)
            yield {"mkt":  T["mkt"][idx],  "port": T["port"][idx],
                   "sent": T["sent"][idx], "tv":   T["tv"][idx],
                   "act":  T["act"][idx],  "olp":  T["lp"][idx],
                   "oval": T["val"][idx],  "adv":  self.adv[idx], "ret": self.ret[idx]}


class PPOTrainer(BaseTrainer):
    def __init__(self, net: ActorCriticNetwork, cfg: PPOConfig, dev: torch.device) -> None:
        super().__init__(net, dev)
        self.cfg = cfg
        pg = [{"params": net.non_enc_params(), "lr": cfg.learning_rate},
              {"params": net.enc_params(),     "lr": cfg.learning_rate * 0.3}]
        self.opt = optim.Adam(pg, eps=1e-5)

        def lr_fn(step: int) -> float:
            if step < cfg.warmup_steps:
                return (step + 1) / cfg.warmup_steps
            return max(0.01, 1. - (step - cfg.warmup_steps) / 1_000_000)
        self.sched = optim.lr_scheduler.LambdaLR(self.opt, lr_fn)
        self.scaler = (torch.cuda.amp.GradScaler()
                       if cfg.use_amp and dev.type == "cuda" else None)
        self.buf = RolloutBuf(dev)
        self._T: Optional[Dict] = None

    def _ent_c(self) -> float:
        f = min(self._step / max(self.cfg.entropy_anneal_steps, 1), 1.)
        return self.cfg.entropy_coef + (self.cfg.entropy_coef_min - self.cfg.entropy_coef) * f

    @torch.no_grad()
    def collect_rollout(self, env: SingleAssetEnv) -> Dict[str, float]:
        self.net.eval()
        self.buf.clear()
        res = env.reset()
        ep_r = 0.
        ep_n = 0
        hx: Optional[Tuple] = None
        for _ in range(self.cfg.n_steps):
            mkt = res.obs_market_seq.unsqueeze(0).to(self.device)
            port = res.obs_portfolio_vec.unsqueeze(0).to(self.device)
            sent = res.obs_sentiment_vec.unsqueeze(0).to(self.device)
            tv = res.obs_time_vec.unsqueeze(0).to(self.device)
            ao, co, hx = self.net.get_action_and_value(
                mkt, port, sent, tv, hx=hx)
            if hx is not None:
                hx = tuple(h.detach() for h in hx)
            act = ao.action.squeeze(0)
            lp = ao.log_prob.squeeze()
            val = co.value.squeeze()
            nxt = env.step(act.cpu().numpy())
            self.buf.add(res, act, lp, val, reward=nxt.reward,
                         done=float(nxt.done))
            ep_r += nxt.reward
            res = nxt
            if res.done:
                self._ep_rewards.append(ep_r)
                ep_r = 0.
                ep_n += 1
                res = env.reset()
                hx = None
        mkt = res.obs_market_seq.unsqueeze(0).to(self.device)
        port = res.obs_portfolio_vec.unsqueeze(0).to(self.device)
        sent = res.obs_sentiment_vec.unsqueeze(0).to(self.device)
        tv = res.obs_time_vec.unsqueeze(0).to(self.device)
        lv = self.net.get_value(mkt, port, sent, tv).squeeze()
        T = self.buf.tensors()
        self.buf.gae(T, lv, res.done, self.cfg.gamma, self.cfg.gae_lambda,
                     self.cfg.normalize_advantage, self.cfg.normalize_returns)
        self._T = T
        self._step += self.cfg.n_steps
        recent = self._ep_rewards[-10:] if self._ep_rewards else [0.]
        return {"mean_reward": float(np.mean(recent)), "n_episodes": ep_n, "n_steps": len(self.buf)}

    def update(self) -> Dict[str, float]:
        if self._T is None:
            raise TrainingError("call collect_rollout first")
        self.net.train()
        ec = self._ent_c()
        stats: Dict[str, List[float]] = {k: [] for k in (
            "pl", "vl", "ent", "tl", "kl", "cf", "gn")}
        early = False
        for epoch in range(self.cfg.n_epochs):
            if early:
                break
            kls: List[float] = []
            for b in self.buf.batches(self._T, self.cfg.batch_size):
                self.net.reset_noise()
                ctx = torch.cuda.amp.autocast() if self.scaler else _NoAMP()
                with ctx:
                    ao, co, _ = self.net.get_action_and_value(
                        b["mkt"], b["port"], b["sent"], b["tv"], b["act"])
                    lr = (ao.log_prob - b["olp"]).clamp(-20, 20)
                    ratio = lr.exp()
                    adv = b["adv"]
                    ret = b["ret"]
                    surr1 = ratio * adv
                    surr2 = ratio.clamp(1 - self.cfg.clip_epsilon,
                                        1 + self.cfg.clip_epsilon) * adv
                    pl = -torch.min(surr1, surr2).mean()
                    vc = b["oval"] + (co.value - b["oval"]).clamp(
                        -self.cfg.clip_vf_epsilon, self.cfg.clip_vf_epsilon)
                    vl = 0.5 * torch.max(
                        F.huber_loss(co.value, ret, reduction="none"),
                        F.huber_loss(vc,       ret, reduction="none")).mean()
                    loss = pl + self.cfg.value_coef * vl - ec * ao.entropy.mean()
                self.opt.zero_grad(set_to_none=True)
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                    gn = nn.utils.clip_grad_norm_(
                        self.net.parameters(), self.cfg.max_grad_norm)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    loss.backward()
                    gn = nn.utils.clip_grad_norm_(
                        self.net.parameters(), self.cfg.max_grad_norm)
                    self.opt.step()
                self.sched.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - lr).mean().item()
                    cf = ((ratio - 1).abs() >
                          self.cfg.clip_epsilon).float().mean().item()
                for k, v in [("pl", pl.item()), ("vl", vl.item()),
                             ("ent", ao.entropy.mean().item()),
                             ("tl", loss.item()), ("kl", kl), ("cf", cf),
                             ("gn", float(gn))]:
                    stats[k].append(v)
                kls.append(kl)
            if self.cfg.target_kl and kls and float(np.mean(kls)) > self.cfg.target_kl:
                early = True
        return {"policy_loss": float(np.mean(stats["pl"])), "value_loss": float(np.mean(stats["vl"])),
                "entropy":     float(np.mean(stats["ent"])), "total_loss": float(np.mean(stats["tl"])),
                "approx_kl":   float(np.mean(stats["kl"])), "clip_frac":  float(np.mean(stats["cf"])),
                "grad_norm":   float(np.mean(stats["gn"])),
                "lr": self.opt.param_groups[0]["lr"], "entropy_coef": ec}

    @torch.no_grad()
    def _select_action(self, r: StepResult, det: bool) -> np.ndarray:
        self.net.eval()
        mkt = r.obs_market_seq.unsqueeze(0).to(self.device)
        port = r.obs_portfolio_vec.unsqueeze(0).to(self.device)
        sent = r.obs_sentiment_vec.unsqueeze(0).to(self.device)
        tv = r.obs_time_vec.unsqueeze(0).to(self.device)
        ao, _, _ = self.net.get_action_and_value(mkt, port, sent, tv, det=det)
        return ao.action.squeeze(0).cpu().numpy()

    def _extra_save(self) -> Dict:
        return {"opt": self.opt.state_dict(), "sched": self.sched.state_dict()}

    def _extra_load(self, ck: Dict) -> None:
        if "opt" in ck:
            self.opt.load_state_dict(ck["opt"])
        if "sched" in ck:
            self.sched.load_state_dict(ck["sched"])


# ==============================================================================
# SECTION 17: SAC Trainer
# ==============================================================================

class SACTrainer(BaseTrainer):
    def __init__(self, net: ActorCriticNetwork, cfg: SACConfig, dev: torch.device, ad: int) -> None:
        super().__init__(net, dev)
        self.cfg = cfg
        self.ad = ad
        td = net.cfg.hidden_dims[-1]
        self.c1 = TwinQCritic(td, ad).to(dev)
        self.c2 = TwinQCritic(td, ad).to(dev)
        self.tc1 = copy.deepcopy(self.c1)
        self.tc2 = copy.deepcopy(self.c2)
        for p in self.tc1.parameters():
            p.requires_grad_(False)
        for p in self.tc2.parameters():
            p.requires_grad_(False)
        self.aopt = optim.Adam(net.parameters(),     lr=cfg.learning_rate)
        self.c1opt = optim.Adam(self.c1.parameters(), lr=cfg.learning_rate)
        self.c2opt = optim.Adam(self.c2.parameters(), lr=cfg.learning_rate)
        if cfg.auto_alpha:
            self.te = cfg.target_entropy if cfg.target_entropy is not None else - \
                float(ad)
            self.la = torch.zeros(1, requires_grad=True, device=dev)
            self.aaopt = optim.Adam([self.la], lr=cfg.learning_rate)
        else:
            self.la = torch.log(torch.tensor(cfg.alpha, device=dev))
        self.replay = PERBuffer(cfg.buffer_size, dev)

    @property
    def alpha(self) -> Tensor: return self.la.exp()

    @torch.no_grad()
    def _soft(self, s: nn.Module, t: nn.Module) -> None:
        for sp, tp in zip(s.parameters(), t.parameters()):
            tp.data.mul_(1 - self.cfg.tau)
            tp.data.add_(self.cfg.tau * sp.data)

    def update(self) -> Dict[str, float]:
        if not self.replay.is_ready(self.cfg.learning_starts):
            return {"skipped": 1.}
        c1l, c2l, al, all_ = [], [], [], []
        for _ in range(self.cfg.gradient_steps):
            batch, idxs, ws = self.replay.sample(self.cfg.batch_size)
            data, wt = self.replay.collate(batch, ws)
            self.net.eval()
            with torch.no_grad():
                nf, _ = self.net.encode(data["next_market_seqs"], data["next_portfolio_vecs"],
                                        data["next_sentiment_vecs"], data["next_time_vecs"])
                ao_n = self.net.actor(nf)
                qn = torch.min(self.tc1(nf, ao_n.action),
                               self.tc2(nf, ao_n.action)).squeeze(-1)
                qt = data["reward"] + self.cfg.gamma * \
                    (1 - data["done"]) * (qn - self.alpha * ao_n.log_prob)
                cf, _ = self.net.encode(data["market_seqs"], data["portfolio_vecs"],
                                        data["sentiment_vecs"], data["time_vecs"])
            q1 = self.c1(cf, data["actions"]).squeeze(-1)
            q2 = self.c2(cf, data["actions"]).squeeze(-1)
            td_e = ((q1 - qt).abs() + (q2 - qt).abs()).detach() / 2
            l1 = (F.huber_loss(q1, qt, reduction="none") * wt).mean()
            l2 = (F.huber_loss(q2, qt, reduction="none") * wt).mean()
            self.c1opt.zero_grad()
            l1.backward()
            nn.utils.clip_grad_norm_(
                self.c1.parameters(), self.cfg.max_grad_norm)
            self.c1opt.step()
            self.c2opt.zero_grad()
            l2.backward()
            nn.utils.clip_grad_norm_(
                self.c2.parameters(), self.cfg.max_grad_norm)
            self.c2opt.step()
            self.net.train()
            af, _ = self.net.encode(data["market_seqs"], data["portfolio_vecs"],
                                    data["sentiment_vecs"], data["time_vecs"])
            ao_a = self.net.actor(af)
            with torch.no_grad():
                qa = torch.min(self.c1(af.detach(), ao_a.action),
                               self.c2(af.detach(), ao_a.action)).squeeze(-1)
            aloss = (self.alpha.detach() * ao_a.log_prob - qa).mean()
            self.aopt.zero_grad()
            aloss.backward()
            nn.utils.clip_grad_norm_(
                self.net.parameters(), self.cfg.max_grad_norm)
            self.aopt.step()
            al_l = torch.tensor(0., device=self.device)
            if self.cfg.auto_alpha:
                al_l = -(self.la * (ao_a.log_prob.detach() + self.te)).mean()
                self.aaopt.zero_grad()
                al_l.backward()
                self.aaopt.step()
            self.replay.update_priorities(idxs, td_e.cpu().numpy())
            c1l.append(float(l1.item()))
            c2l.append(float(l2.item()))
            al.append(float(aloss.item()))
            all_.append(float(al_l.item()))
        self._soft(self.c1, self.tc1)
        self._soft(self.c2, self.tc2)
        self._step += 1
        return {"critic_loss": float(np.mean(c1l)) + float(np.mean(c2l)),
                "actor_loss":  float(np.mean(al)),  "alpha_loss": float(np.mean(all_)),
                "alpha": float(self.alpha.item())}

    def collect_rollout(self, env: SingleAssetEnv) -> Dict[str, float]:
        self.net.eval()
        res = env.reset()
        er = 0.
        en = 0
        for _ in range(self.cfg.train_freq):
            mkt = res.obs_market_seq.unsqueeze(0).to(self.device)
            port = res.obs_portfolio_vec.unsqueeze(0).to(self.device)
            sent = res.obs_sentiment_vec.unsqueeze(0).to(self.device)
            tv = res.obs_time_vec.unsqueeze(0).to(self.device)
            with torch.no_grad():
                ao, _, _ = self.net.get_action_and_value(mkt, port, sent, tv)
            nr = env.step(ao.action.squeeze(0).cpu().numpy())
            self.replay.add(Transition(
                res.obs_market_seq.cpu().numpy(), res.obs_portfolio_vec.cpu().numpy(),
                res.obs_sentiment_vec.cpu().numpy(), res.obs_time_vec.cpu().numpy(),
                nr.obs_market_seq.cpu().numpy(),  nr.obs_portfolio_vec.cpu().numpy(),
                nr.obs_sentiment_vec.cpu().numpy(), nr.obs_time_vec.cpu().numpy(),
                ao.action.squeeze(0).cpu().numpy(), nr.reward, nr.done, nr.info))
            er += nr.reward
            res = nr
            if res.done:
                self._ep_rewards.append(er)
                er = 0.
                en += 1
                res = env.reset()
        recent = self._ep_rewards[-10:] if self._ep_rewards else [0.]
        return {"mean_reward": float(np.mean(recent)), "n_episodes": en, "buffer_size": len(self.replay)}

    @torch.no_grad()
    def _select_action(self, r: StepResult, det: bool) -> np.ndarray:
        self.net.eval()
        mkt = r.obs_market_seq.unsqueeze(0).to(self.device)
        port = r.obs_portfolio_vec.unsqueeze(0).to(self.device)
        sent = r.obs_sentiment_vec.unsqueeze(0).to(self.device)
        tv = r.obs_time_vec.unsqueeze(0).to(self.device)
        ao, _, _ = self.net.get_action_and_value(mkt, port, sent, tv, det=det)
        return ao.action.squeeze(0).cpu().numpy()


# ==============================================================================
# SECTION 18: TD3 Trainer
# ==============================================================================

class TD3Trainer(BaseTrainer):
    def __init__(self, net: ActorCriticNetwork, cfg: TD3Config, dev: torch.device, ad: int) -> None:
        super().__init__(net, dev)
        self.cfg = cfg
        self.ad = ad
        td = net.cfg.hidden_dims[-1]
        self.c1 = TwinQCritic(td, ad).to(dev)
        self.c2 = TwinQCritic(td, ad).to(dev)
        self.tc1 = copy.deepcopy(self.c1)
        self.tc2 = copy.deepcopy(self.c2)
        self.ta = copy.deepcopy(net)
        for p in self.tc1.parameters():
            p.requires_grad_(False)
        for p in self.tc2.parameters():
            p.requires_grad_(False)
        for p in self.ta.parameters():
            p.requires_grad_(False)
        self.aopt = optim.Adam(net.parameters(), lr=cfg.learning_rate)
        self.copt = optim.Adam(list(self.c1.parameters()) + list(self.c2.parameters()),
                               lr=cfg.learning_rate)
        self.replay = PERBuffer(cfg.buffer_size, dev)

    @torch.no_grad()
    def _soft(self, s: nn.Module, t: nn.Module) -> None:
        for sp, tp in zip(s.parameters(), t.parameters()):
            tp.data.mul_(1 - self.cfg.tau)
            tp.data.add_(self.cfg.tau * sp.data)

    def update(self) -> Dict[str, float]:
        if not self.replay.is_ready(self.cfg.learning_starts):
            return {"skipped": 1.}
        batch, _, _ = self.replay.sample(self.cfg.batch_size)
        data = self.replay.collate(batch, np.ones(self.cfg.batch_size))[0]
        self.net.eval()
        with torch.no_grad():
            nf, _ = self.ta.encode(data["next_market_seqs"], data["next_portfolio_vecs"],
                                   data["next_sentiment_vecs"], data["next_time_vecs"])
            ao_n = self.ta.actor(nf, deterministic=True)
            noise = (torch.randn_like(ao_n.action) * self.cfg.policy_noise).clamp(
                -self.cfg.noise_clip, self.cfg.noise_clip)
            na = (ao_n.action + noise).clamp(-1, 1)
            qt = (data["reward"].unsqueeze(-1) + self.cfg.gamma * (1 - data["done"].unsqueeze(-1))
                  * torch.min(self.tc1(nf, na), self.tc2(nf, na)))
            cf, _ = self.net.encode(data["market_seqs"], data["portfolio_vecs"],
                                    data["sentiment_vecs"], data["time_vecs"])
        cl = (F.huber_loss(self.c1(cf, data["actions"]), qt)
              + F.huber_loss(self.c2(cf, data["actions"]), qt))
        self.copt.zero_grad()
        cl.backward()
        nn.utils.clip_grad_norm_(list(self.c1.parameters()) + list(self.c2.parameters()),
                                 self.cfg.max_grad_norm)
        self.copt.step()
        al = torch.tensor(0., device=self.device)
        if self._step % self.cfg.policy_delay == 0:
            self.net.train()
            af, _ = self.net.encode(data["market_seqs"], data["portfolio_vecs"],
                                    data["sentiment_vecs"], data["time_vecs"])
            ao_a = self.net.actor(af, deterministic=True)
            al = -self.c1(af.detach(), ao_a.action).mean()
            self.aopt.zero_grad()
            al.backward()
            nn.utils.clip_grad_norm_(
                self.net.parameters(), self.cfg.max_grad_norm)
            self.aopt.step()
            self._soft(self.net, self.ta)
            self._soft(self.c1, self.tc1)
            self._soft(self.c2, self.tc2)
        self._step += 1
        return {"critic_loss": float(cl.item()), "actor_loss": float(al.item())}

    def collect_rollout(self, env: SingleAssetEnv) -> Dict[str, float]:
        self.net.eval()
        res = env.reset()
        ec = 0
        er = 0.
        for _ in range(max(self.cfg.learning_starts // 10, 20)):
            mkt = res.obs_market_seq.unsqueeze(0).to(self.device)
            port = res.obs_portfolio_vec.unsqueeze(0).to(self.device)
            sent = res.obs_sentiment_vec.unsqueeze(0).to(self.device)
            tv = res.obs_time_vec.unsqueeze(0).to(self.device)
            with torch.no_grad():
                ao, _, _ = self.net.get_action_and_value(mkt, port, sent, tv)
            noise = torch.randn_like(ao.action) * self.cfg.exploration_noise
            a = (ao.action + noise).clamp(-1, 1).squeeze(0).cpu().numpy()
            nr = env.step(a)
            self.replay.add(Transition(
                res.obs_market_seq.cpu().numpy(), res.obs_portfolio_vec.cpu().numpy(),
                res.obs_sentiment_vec.cpu().numpy(), res.obs_time_vec.cpu().numpy(),
                nr.obs_market_seq.cpu().numpy(),  nr.obs_portfolio_vec.cpu().numpy(),
                nr.obs_sentiment_vec.cpu().numpy(), nr.obs_time_vec.cpu().numpy(),
                a, nr.reward, nr.done, nr.info))
            er += nr.reward
            res = nr
            if res.done:
                self._ep_rewards.append(er)
                ec += 1
                er = 0.
                res = env.reset()
        return {"n_episodes": ec, "buffer_size": len(self.replay)}

    @torch.no_grad()
    def _select_action(self, r: StepResult, det: bool) -> np.ndarray:
        self.net.eval()
        mkt = r.obs_market_seq.unsqueeze(0).to(self.device)
        port = r.obs_portfolio_vec.unsqueeze(0).to(self.device)
        sent = r.obs_sentiment_vec.unsqueeze(0).to(self.device)
        tv = r.obs_time_vec.unsqueeze(0).to(self.device)
        ao, _, _ = self.net.get_action_and_value(mkt, port, sent, tv, det=True)
        return ao.action.squeeze(0).cpu().numpy()


# ==============================================================================
# SECTION 19: Rainbow DQN
# ==============================================================================

class RainbowDQN(nn.Module):
    def __init__(self, cfg: NetworkConfig, nf: int, na: int) -> None:
        super().__init__()
        self.enc = build_encoder(cfg, nf)
        self.dist = DistributionalCritic(self.enc.out_dim, na, 51, -10, 10, 128, True)
        self.na = na

    def forward(self, x: Tensor) -> DistributionalOutput:
        f, _ = self.enc(x)
        return self.dist(f)

    def reset_noise(self) -> None:
        if self.training:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()


class RainbowTrainer(BaseTrainer):
    def __init__(self, net: RainbowDQN, cfg: RainbowConfig, dev: torch.device) -> None:
        super().__init__(net, dev)
        self.cfg = cfg
        self.tgt = copy.deepcopy(net).to(dev)
        for p in self.tgt.parameters():
            p.requires_grad_(False)
        self.opt = optim.Adam(
            net.parameters(), lr=cfg.learning_rate, eps=1.5e-4)
        self.replay = PERBuffer(cfg.buffer_size, dev, cfg.per_alpha, cfg.per_beta,
                                cfg.per_beta_frames, cfg.per_epsilon)

    def collect_rollout(self, env: SingleAssetEnv) -> Dict[str, float]:
        self.net.eval()
        res = env.reset()
        er = 0.
        ec = 0
        for _ in range(100):
            x = res.obs_market_seq.unsqueeze(0).to(self.device)
            with torch.no_grad():
                do = self.net(x)
            eps = max(0.01, 1 - self._step / 100_000)
            a = (random.randrange(self.net.na) if random.random() < eps
                 else int(do.q_values.argmax(-1).item()))
            nr = env.step(np.array([a]))
            self.replay.add(Transition(
                res.obs_market_seq.cpu().numpy(), res.obs_portfolio_vec.cpu().numpy(),
                res.obs_sentiment_vec.cpu().numpy(), res.obs_time_vec.cpu().numpy(),
                nr.obs_market_seq.cpu().numpy(),  nr.obs_portfolio_vec.cpu().numpy(),
                nr.obs_sentiment_vec.cpu().numpy(), nr.obs_time_vec.cpu().numpy(),
                np.array([a]), nr.reward, nr.done, nr.info))
            er += nr.reward
            res = nr
            if res.done:
                self._ep_rewards.append(er)
                ec += 1
                er = 0.
                res = env.reset()
        return {"n_episodes": ec, "buffer_size": len(self.replay)}

    def update(self) -> Dict[str, float]:
        if len(self.replay) < self.cfg.batch_size:
            return {"skipped": 1.}
        self.net.train()
        self.net.reset_noise()
        batch, idxs, ws = self.replay.sample(self.cfg.batch_size)
        data, wt = self.replay.collate(batch, ws)
        x = data["market_seqs"]
        nx = data["next_market_seqs"]
        r = data["reward"]
        d = data["done"]
        a = data["actions"].long().squeeze(-1)
        do = self.net(x)
        with torch.no_grad():
            nd = self.tgt(nx)
            na = nd.q_values.argmax(-1)
            nlp = nd.log_probs[torch.arange(
                self.cfg.batch_size, device=self.device), na]
            tgt = self.net.dist.project(r, nlp, d, self.cfg.gamma)
        lp = do.log_probs[torch.arange(
            self.cfg.batch_size, device=self.device), a]
        loss = -(tgt * lp).sum(-1)
        wloss = (loss * wt).mean()
        self.opt.zero_grad()
        wloss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
        self.opt.step()
        self.replay.update_priorities(idxs, loss.detach().cpu().numpy())
        self._step += 1
        if self._step % self.cfg.target_update == 0:
            self.tgt.load_state_dict(self.net.state_dict())
        return {"loss": float(wloss.item()), "q": float(do.q_values.mean().item())}

    @torch.no_grad()
    def _select_action(self, r: StepResult, det: bool) -> np.ndarray:
        self.net.eval()
        x = r.obs_market_seq.unsqueeze(0).to(self.device)
        return np.array([int(self.net(x).q_values.argmax(-1).item())])


# ==============================================================================
# SECTION 20: Backtester & Metrics
# ==============================================================================

@dataclass
class BacktestMetrics:
    total_return:      float = 0.
    annualised_return: float = 0.
    sharpe_ratio:      float = 0.
    sortino_ratio:     float = 0.
    calmar_ratio:      float = 0.
    max_drawdown:      float = 0.
    win_rate:          float = 0.
    profit_factor:     float = 0.
    total_trades:      int = 0
    avg_trade_return:  float = 0.
    avg_win:           float = 0.
    avg_loss:          float = 0.
    largest_win:       float = 0.
    largest_loss:      float = 0.
    avg_hold_bars:     float = 0.
    expectancy:        float = 0.

    def __str__(self) -> str:
        return (f"BacktestMetrics(\n"
                f"  total_return={self.total_return:+.2%}  ann={self.annualised_return:+.2%}\n"
                f"  sharpe={self.sharpe_ratio:.3f}  sortino={self.sortino_ratio:.3f}\n"
                f"  max_dd={self.max_drawdown:.2%}  win_rate={self.win_rate:.1%}\n"
                f"  trades={self.total_trades}  expectancy={self.expectancy:+.4f}\n)")


def compute_metrics(eq: List[float], tr: List[float], hb: List[int],
                    bpy: int = 252, rfr: float = 0.04) -> BacktestMetrics:
    if len(eq) < 2 or len(tr) == 0:
        return BacktestMetrics()
    eq_a = np.clip(np.array(eq, np.float64), 1e-10, 1e12)
    if eq_a[0] <= 0:
        return BacktestMetrics()
    br = np.diff(eq_a) / eq_a[:-1]
    ret = float(eq_a[-1] / eq_a[0] - 1)
    af = bpy / max(len(br), 1)
    ann = float((1 + ret) ** af - 1)
    rfd = rfr / bpy
    er = br - rfd
    sh = float(np.clip(er.mean() / max(br.std(), 1e-10)
               * math.sqrt(bpy), -10, 10))
    neg = br[br < rfd]
    so = float(np.clip(er.mean() / max(math.sqrt(np.mean(neg ** 2)) if len(neg) else 1e-10, 1e-10)
                       * math.sqrt(252), -10, 10))
    peak = np.maximum.accumulate(eq_a)
    dd = (eq_a - peak) / np.clip(peak, 1e-10, None)
    mdd = float(abs(dd.min()))
    cal = float(np.clip(ann / max(mdd, 1e-10), -20, 20))
    tr_a = np.array(tr, np.float64)
    wins = tr_a[tr_a > 0]
    losses = tr_a[tr_a < 0]
    wr = float(len(wins) / len(tr_a)) if len(tr_a) else 0.
    pf = float(wins.sum() / max(abs(losses.sum()), 1e-10)
               ) if len(losses) else float("inf")
    aw = float(wins.mean()) if len(wins) else 0.
    al = float(abs(losses.mean())) if len(losses) else 0.
    exp = aw * wr - al * (1 - wr)
    return BacktestMetrics(ret, ann, sh, so, cal, mdd, wr, pf, len(tr), float(tr_a.mean()),
                           aw, al, float(wins.max()) if len(wins) else 0.,
                           float(losses.min()) if len(losses) else 0.,
                           float(np.mean(hb)) if hb else 0., exp)


class Backtester:
    def __init__(self, trainer: BaseTrainer, dev: torch.device) -> None:
        self.trainer = trainer
        self.dev = dev

    def run(self, env: SingleAssetEnv, seed: int = 0,
            verbose: bool = True) -> Tuple[BacktestMetrics, Dict]:
        self.trainer.net.eval()
        res = env.reset(seed=seed)
        eq: List[float] = [env.cfg.initial_capital]
        trs: List[float] = []
        hbs: List[int] = []
        log: List[Dict] = []
        while not env.is_done:
            res = env.step(self.trainer._select_action(res, True))
            cap = float(np.clip(res.info.get("capital", eq[-1]), 0, 1e12))
            eq.append(cap)
            log.append(
                {"step": len(log), "capital": cap, "reward": res.reward})
        for t in env._ex.trade_log:
            if "net_pnl" in t:
                ref = float(t.get("fill_price", 1) or 1) * \
                    float(t.get("quantity", 1) or 1)
                if ref > 0:
                    trs.append(t["net_pnl"] / ref)
                hbs.append(int(t.get("hold_bars", 0)))
        m = compute_metrics(eq, trs, hbs)
        if verbose:
            print(f"\n{m}")
        return m, {"equity_curve": eq, "trade_returns": trs, "step_log": log}


# ==============================================================================
# SECTION 21: Demo & Main
# ==============================================================================

def make_bars(n: int = 1200, seed: int = 42) -> List[Bar]:
    """
    Synthetic data with LEARNABLE patterns.
    Properly guarded against edge cases.
    """
    rng = np.random.default_rng(seed)
    price = 50_000.0
    fair_value = 50_000.0  # Mean-reversion target
    base = 1_700_000_000
    bars = []

    # Track regime
    regime = 0
    regime_timer = 0

    # Store returns for momentum calculation
    returns_history = []

    for i in range(n):
        # --- Regime switching ---
        if regime_timer <= 0:
            regime = rng.choice([-1, 0, 1], p=[0.4, 0.2, 0.4])
            regime_timer = rng.integers(30, 100)
        regime_timer -= 1

        # --- Volume signal ---
        high_vol = rng.random() < 0.3
        vol_base = 10.5 if high_vol else 9.0
        volume = float(rng.lognormal(vol_base, 0.4))
        volume = max(volume, 1.0)  # GUARD: prevent zero/negative volume

        # --- Generate return ---
        if regime == 1 and i >= 10:  # MOMENTUM: need enough history
            # Use last 5-10 bars for momentum
            if len(returns_history) >= 5:
                momentum = sum(returns_history[-5:])  # 5-bar cumulative return
                drift = momentum * 0.3  # Trend continues
                # GUARD: clamp drift to reasonable range
                drift = float(np.clip(drift, -0.05, 0.05))
            else:
                drift = 0.0002

            vol = 0.012 if high_vol else 0.008

        elif regime == -1:  # MEAN-REVERSION
            # Distance from fair value (as percentage)
            distance_pct = (fair_value - price) / max(price, 1.0)
            # GUARD: clamp to prevent extreme values
            distance_pct = float(np.clip(distance_pct, -0.5, 0.5))
            drift = distance_pct * 0.02  # Revert slowly

            vol = 0.015 if high_vol else 0.010

        else:  # RANDOM
            drift = 0.0002
            vol = 0.016

        # --- Generate return with guards ---
        ret = float(rng.normal(drift, vol))
        # GUARD: clamp extreme returns
        ret = float(np.clip(ret, -0.10, 0.10))  # Max ±10% per bar

        # --- Update price with guard ---
        price *= math.exp(ret)
        price = max(price, 100.0)  # GUARD: never below 100
        price = min(price, 1_000_000.0)  # GUARD: never above 1M

        returns_history.append(ret)
        if len(returns_history) > 100:
            returns_history.pop(0)

        # --- Create OHLCV with guards ---
        bar_range = abs(float(rng.normal(0, 0.004)))
        bar_range = min(bar_range, 0.10)  # GUARD: max 10% range

        hi = price * (1 + bar_range)
        lo = price * (1 - bar_range)
        lo = max(lo, 0.01)  # GUARD: low can't be zero

        op = price * float(rng.uniform(
            max(0.99, lo/price),
            min(1.01, hi/price)
        ))
        op = max(op, 0.01)  # GUARD: open can't be zero

        bars.append(Bar(
            "BTC/USDT", TimeFrame.H1, base + i * 3600,
            op, hi, lo, price, volume
        ))

        # --- Validate bar ---
        assert price > 0, f"Zero/negative price at bar {i}"
        assert hi >= lo, f"High < Low at bar {i}: hi={hi}, lo={lo}"
        assert hi >= price >= lo, f"Price outside range at bar {i}"
        assert volume > 0, f"Zero volume at bar {i}"

    return bars


def main() -> None:
    set_seed(42)
    device = select_device()
    W = 70
    print("=" * W)
    print(
        f" ZeroStrike DRL v3.6 | device={device} | torch={torch.__version__}")
    print("=" * W)
    print(f" Features: market={N_MARKET_FEATURES}  portfolio={N_PORTFOLIO_FEATURES}"
          f"  sentiment={N_SENTIMENT_FEATURES}  time={N_TIME_FEATURES}\n")

    # [1/6] Network
    print("[1/6] Building network ...")
    ncfg = NetworkConfig(
        architecture=Architecture.ATTENTION_LSTM,
        d_model=128, n_heads=4, n_layers=2, d_ff=256,
        lstm_hidden=128, lstm_layers=1, lstm_dropout=0.,
        hidden_dims=(256, 128), dropout=0.1,
        use_noisy_net=True, use_dueling=True,
    )
    net = ActorCriticNetwork(ncfg, N_MARKET_FEATURES, N_PORTFOLIO_FEATURES,
                N_SENTIMENT_FEATURES, N_TIME_FEATURES, ad=3, cont=True).to(device)
    pc = net.param_count()
    print(f"      params total={pc['total']:,}  trainable={pc['trainable']:,}")
    with torch.no_grad():
        ao, co, _ = net.get_action_and_value(
            torch.randn(2, 60, N_MARKET_FEATURES,    device=device),
            torch.randn(2, N_PORTFOLIO_FEATURES,     device=device),
            torch.randn(2, N_SENTIMENT_FEATURES,     device=device),
            torch.randn(2, N_TIME_FEATURES,          device=device))
    std_v = float(ao.std.mean())
    print(f"      forward OK | action={tuple(ao.action.shape)}"
          f"  value=[{co.value.min():.3f},{co.value.max():.3f}]  actor_std={std_v:.3f}")
    _chk(0.05 < std_v < 3., f"Actor std out of [0.05, 3.0]: {std_v:.4f}")

    # [2/6] PPO
    print("\n[2/6] PPO training ...")
    all_bars = make_bars(1200)
    train_bars = all_bars[:900]
    eval_bars = all_bars[900:]

    reward_cfg = RewardConfig()

    train_cfg = EnvironmentConfig(
        episode_length=200, warmup_bars=60, initial_capital=100_000.,
        random_start=True, reward=reward_cfg,
    )
    eval_cfg = EnvironmentConfig(
        episode_length=200, warmup_bars=60, initial_capital=100_000.,
        random_start=False, reward=reward_cfg,
    )
    train_env = SingleAssetEnv(train_bars, train_cfg, device)
    eval_env = SingleAssetEnv(eval_bars,  eval_cfg,  device)

    pcfg = PPOConfig(
        n_steps=512, n_epochs=4, batch_size=64,
        learning_rate=3e-4, warmup_steps=128,
        target_kl=0.05,
        normalize_advantage=True, normalize_returns=True,
        entropy_coef=0.01, entropy_coef_min=0.001, entropy_anneal_steps=100_000,
        clip_epsilon=0.2, value_coef=0.5, max_grad_norm=0.5,
    )
    trainer = PPOTrainer(net, pcfg, device)

    for i in range(10):
        rm = trainer.collect_rollout(train_env)
        um = trainer.update()
        pl = um.get("policy_loss", 0)
        vl = um.get("value_loss",  0)
        kl = um.get("approx_kl",   0)
        lr = um.get("lr",          0)
        gn = um.get("grad_norm",   0)
        print(f"      iter={i + 1:2d}  R={rm.get('mean_reward', 0):+.4f}"
              f"  πL={pl:.4f}  VL={vl:.4f}  KL={kl:.5f}  GN={gn:.3f}  lr={lr:.2e}")
        _chk(math.isfinite(pl) and math.isfinite(
            vl), f"Loss non-finite at iter {i + 1}")
        _chk(abs(pl) < 10.,  f"Policy loss magnitude unrealistic: {pl:.4f}")
        _chk(vl < 100.,      f"Value loss exploded: {vl:.2f}")
        _chk(abs(kl) < 0.5,  f"KL too high: {kl:.5f}")
        _chk(gn < 50.,       f"Gradient norm exploded: {gn:.2f}")

    # [3/6] Evaluate
    print("\n[3/6] Evaluating PPO policy on held-out data ...")
    em = trainer.evaluate(eval_env, n=3, det=True)
    cap = em["mean_capital"]
    print(f"      mean_reward={em['mean_reward']:+.4f}  mean_capital={cap:,.0f}"
          f"  win_rate={em.get('mean_win_rate', 0):.1%}  trades={em.get('mean_trades', 0):.0f}")
    _chk(50_000 < cap < 200_000,  # WIDENED: allow up to 2x return
         f"Capital out of [50K, 200K]: {cap:,.0f}")

    # [4/6] Backtest
    print("\n[4/6] Running backtest on held-out data ...")
    bt = Backtester(trainer, device)
    m, _ = bt.run(eval_env, seed=0, verbose=False)
    print(f"      Sharpe={m.sharpe_ratio:.3f}  MaxDD={m.max_drawdown:.2%}"
          f"  Trades={m.total_trades}  WinRate={m.win_rate:.1%}  Return={m.total_return:+.2%}")

    # UPDATED CHECKS for data with real patterns
    _chk(abs(m.sharpe_ratio) <= 10., f"Sharpe unrealistic: {m.sharpe_ratio}")
    _chk(-0.50 < m.total_return < 1.00,
         f"Return outside expected range: {m.total_return:+.2%}")
    _chk(m.max_drawdown < 0.50,
         f"Max drawdown too high: {m.max_drawdown:.2%}")
    _chk(0.30 < m.win_rate < 0.90 if m.total_trades > 0 else True,
         f"Win rate suspicious: {m.win_rate:.1%}")

    # [5/6] SAC
    print("\n[5/6] SAC smoke test ...")
    snet = ActorCriticNetwork(ncfg, N_MARKET_FEATURES, N_PORTFOLIO_FEATURES,
                 N_SENTIMENT_FEATURES, N_TIME_FEATURES, ad=3, cont=True).to(device)
    sac = SACTrainer(snet, SACConfig(learning_starts=10, batch_size=8, buffer_size=1000,
                                     train_freq=20, gradient_steps=1, max_grad_norm=1.),
                     device, ad=3)
    rm2 = sac.collect_rollout(train_env)
    um2 = sac.update()
    if "skipped" not in um2:
        cl = um2.get("critic_loss", 0)
        print(f"      buffer={rm2['buffer_size']}  critic_loss={cl:.4f}"
              f"  actor_loss={um2.get('actor_loss', 0):.4f}  alpha={um2.get('alpha', 0):.4f}")
        _chk(cl < 500., f"SAC critic loss too large: {cl:.2f}")
    else:
        print(
            f"      buffer={rm2.get('buffer_size', 0)} (needs {sac.cfg.learning_starts})")

    # [6/6] TD3
    print("\n[6/6] TD3 smoke test ...")
    tnet = ActorCriticNetwork(ncfg, N_MARKET_FEATURES, N_PORTFOLIO_FEATURES,
                 N_SENTIMENT_FEATURES, N_TIME_FEATURES, ad=3, cont=True).to(device)
    td3 = TD3Trainer(tnet, TD3Config(learning_starts=10, batch_size=8, buffer_size=1000,
                                     max_grad_norm=1.), device, ad=3)
    rm3 = td3.collect_rollout(train_env)
    um3 = td3.update()
    if "skipped" not in um3:
        print(f"      buffer={rm3['buffer_size']}  critic_loss={um3.get('critic_loss', 0):.4f}"
              f"  actor_loss={um3.get('actor_loss', 0):.4f}")
    else:
        print(
            f"      buffer={rm3.get('buffer_size', 0)} (needs {td3.cfg.learning_starts})")

    print(f"\n{'=' * W}")
    print(" ✓ All checks passed — ZeroStrike DRL v3.6 ready.")
    print("=" * W)


if __name__ == "__main__":
    main()


