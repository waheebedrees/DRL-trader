from __future__ import annotations


import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any


from state import (
    PositionState,
    PortfolioSnapshot,
    Side,
    PortfolioSnapshot,
    PositionState,
)


@dataclass
class ExecutionConfig:
    """Execution simulation parameters."""
    commission_rate:      float = 0.001
    slippage_rate:        float = 0.0005
    funding_rate_per_bar: float = 0.00005
    max_leverage:         float = 3.0
    min_order_notional:   float = 10.0


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
