

from __future__ import annotations


import numpy as np
from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple, Any
import copy
import torch




from single_asset_env import SingleAssetEnv
from base_trainer import BaseTrainer

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
