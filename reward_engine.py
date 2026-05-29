# reward_engine.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import RewardConfig


@dataclass
class StepContext:
    realised_pnl:       float
    realised_pnl_pct:   float
    total_pnl_pct:      float
    unrealised_pnl_pct: float
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


class RollingMetrics(nn.Module):

    def __init__(self, W: int, rfr: float, dev: torch.device) -> None:
        super().__init__()
        self.W = W
        self.rfr = rfr
        self.register_buffer("_r",    torch.zeros(W, device=dev))
        self.register_buffer("_ptr",  torch.zeros(
            1, dtype=torch.int64, device=dev))
        self.register_buffer("_cnt",  torch.zeros(
            1, dtype=torch.int64, device=dev))
        self.register_buffer("_pk",   torch.zeros(1, device=dev))
        self.register_buffer("_wins", torch.zeros(
            1, dtype=torch.int64, device=dev))
        self.register_buffer("_loss", torch.zeros(
            1, dtype=torch.int64, device=dev))

    def reset(self) -> None:
        for b in (self._r, self._ptr, self._cnt,
                  self._pk, self._wins, self._loss):
            b.zero_()

    @torch.no_grad()
    def push_ret(self, r: float) -> None:
        r = float(np.clip(r, -0.5, 0.5))
        idx = int(self._ptr.item())
        self._r[idx] = r
        self._ptr[0] = (idx + 1) % self.W
        self._cnt[0] = min(int(self._cnt.item()) + 1, self.W)

    @torch.no_grad()
    def push_peak(self, cap: float) -> None:
        v = float(np.clip(cap, 0, 1e12))
        if v > float(self._pk.item()):
            self._pk[0] = v

    @torch.no_grad()
    def push_trade(self, pnl: float) -> None:
        if pnl > 0:
            self._wins[0] += 1
        else:
            self._loss[0] += 1

    def _active(self) -> torch.Tensor:
        n = int(self._cnt.item())
        return self._r[:max(1, min(n, self.W))]

    @torch.no_grad()
    def sharpe(self) -> float:
        r = self._active()
        if len(r) < 20:
            return 0.0
        ex = r.mean().item() - self.rfr / 252
        std = r.std().clamp(1e-8).item()
        return float(np.tanh(ex / std * math.sqrt(252) / 3))

    @torch.no_grad()
    def dd_penalty(self, dd: float) -> float:
        dd = float(np.clip(dd, 0, 1))
        if dd <= 0.10:          # free zone — normal fluctuation
            return 0.0
        if dd <= 0.20:
            return -(dd - 0.10) * 1.0
        return -(0.10 + (dd - 0.20) ** 2 * 4.0)

    @torch.no_grad()
    def scale_pnl(
        self,
        pct: float,
        round_trip_cost: float = 0.003,
    ) -> float:
        pct = float(np.clip(pct, -0.5, 0.5))
        net = (pct - round_trip_cost) if pct > 0 else (pct + round_trip_cost)
        net = float(np.clip(net, -0.5, 0.5))
        net_pp = net * 100.0
        return math.copysign(math.log1p(abs(net_pp) * 2.0), net_pp)

    @torch.no_grad()
    def trade_quality(self) -> float:
        tot = float((self._wins + self._loss).item())
        if tot < 5:
            return 0.0
        wr = float(self._wins.item()) / tot
        return float(np.tanh((wr - 0.5) * 4))


class RewardEngine(nn.Module):

    def __init__(self, cfg: RewardConfig, dev: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.dev = dev
        self.m = RollingMetrics(cfg.returns_window, cfg.risk_free_rate, dev)

    def reset(self) -> None:
        self.m.reset()

    @torch.no_grad()
    def step(
        self,
        ctx: StepContext,
        position_size_pct: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        c = self.cfg
        self.m.push_ret(ctx.total_pnl_pct)
        self.m.push_peak(ctx.portfolio_value)
        if ctx.closed:
            self.m.push_trade(ctx.realised_pnl)

        comps: Dict[str, float] = {}

        # ── PnL ──────────────────────────────────────────────────────────────
        if ctx.closed:
            comps["pnl"] = self.m.scale_pnl(ctx.realised_pnl_pct) * c.w_pnl
        elif ctx.position_side != 0:
            # MTM: light continuous signal, no cost deduction
            comps["pnl_mtm"] = (
                self.m.scale_pnl(ctx.unrealised_pnl_pct, round_trip_cost=0.0)
                * c.w_pnl * 0.10
            )

        # ── Risk ─────────────────────────────────────────────────────────────
        comps["sh"] = self.m.sharpe() * c.w_sharpe
        comps["dd"] = self.m.dd_penalty(ctx.drawdown) * c.w_drawdown
        comps["tq"] = self.m.trade_quality() * c.w_trade_quality

        # ── Bonuses ───────────────────────────────────────────────────────────
        if ctx.hit_tp:
            comps["tp"] = c.bonus_hit_tp * max(1, ctx.tp_level)
        if ctx.hit_sl:
            comps["sl"] = c.penalty_hit_sl

        if (not ctx.closed
                and ctx.position_side != 0
                and ctx.unrealised_pnl_pct > c.hold_profit_threshold):
            comps["hold"] = c.bonus_hold_profit

        if ((ctx.position_side == 1 and ctx.price_return > 0) or
                (ctx.position_side == -1 and ctx.price_return < 0)):
            comps["dir"] = c.bonus_correct_direction

        # ── Penalties ─────────────────────────────────────────────────────────
        if ctx.drawdown > c.max_drawdown_threshold:
            excess = ctx.drawdown - c.max_drawdown_threshold
            comps["ddb"] = c.penalty_large_drawdown * (1.0 + excess * 3.0)

        if ctx.daily_trades > c.max_daily_trades:
            comps["ot"] = c.penalty_overtrade

        if ctx.leverage > c.max_leverage:
            comps["lv"] = c.penalty_over_leverage

        if ctx.hold_bars > c.max_hold_bars:
            comps["hld"] = c.penalty_hold_too_long

        raw = sum(comps.values())
        total = float(np.clip(raw, -c.clip_reward,
                      c.clip_reward)) * c.reward_scale
        return total, comps
