# diagnostics.py  — CREATE THIS FILE
"""
Debugging and monitoring tools.
Import in main() to track what the agent is actually learning.
"""

from __future__ import annotations
import os

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════
# CLASS 1: RewardDebugger
# Purpose: detect reward hacking and sparse signals
# USE IN: training loop, log every step
# ══════════════════════════════════════════════════════

class RewardDebugger:
    """
    Tracks reward components to detect:
    - Reward hacking (one component > 70%)
    - Sparse signals (component rarely fires)
    - Sign flips (reward bouncing positive/negative)

    USAGE in training loop:
        debugger = RewardDebugger()
        # inside collect_rollout:
        if 'reward_components' in nxt.info:
            debugger.log(nxt.info['reward_components'], nxt.reward)
        # every eval interval:
        print(debugger.report())
        debugger.check_hacking()   # raises warning if needed
    """

    def __init__(self, window: int = 200) -> None:
        self.window = window
        self.comps:  Dict[str, List[float]] = defaultdict(list)
        self.totals: List[float] = []
        self.ep_rewards: List[float] = []

    def log(self, components: Dict[str, float], total: float) -> None:
        for k, v in components.items():
            self.comps[k].append(float(v))
        self.totals.append(float(total))

    def dominance(self) -> Dict[str, float]:
        """Fraction of total reward mass each component carries."""
        if not self.comps:
            return {}
        result = {}
        for k, vs in self.comps.items():
            recent = vs[-self.window:]
            result[k] = float(np.mean(np.abs(recent))) if recent else 0.0
        total = sum(result.values()) + 1e-10
        return {k: v / total for k, v in result.items()}

    def check_hacking(self) -> List[str]:
        """Return list of components that dominate > 65%."""
        dom = self.dominance()
        return [f"{k}({v:.0%})" for k, v in dom.items() if v > 0.65]

    def sparsity(self) -> Dict[str, float]:
        """Fraction of steps where each component is non-zero."""
        result = {}
        for k, vs in self.comps.items():
            recent = vs[-self.window:]
            result[k] = float(np.mean([abs(v) > 1e-6 for v in recent]))
        return result

    def report(self) -> str:
        dom = self.dominance()
        spar = self.sparsity()
        hacks = self.check_hacking()

        lines = ["\n📊 REWARD DEBUGGER", "=" * 45]

        # Component table
        lines.append(f"  {'comp':<10} {'dominance':>10} {'sparsity':>10}")
        lines.append(f"  {'-'*10} {'-'*10} {'-'*10}")
        for k in sorted(dom, key=dom.get, reverse=True):
            bar = "█" * int(dom[k] * 15)
            lines.append(
                f"  {k:<10} {dom[k]:>9.1%}  {spar.get(k, 0):>9.1%}  {bar}"
            )

        # Summary stats
        if self.totals:
            recent = self.totals[-self.window:]
            lines.append(f"\n  Total reward:  mean={np.mean(recent):+.4f}"
                         f"  std={np.std(recent):.4f}")

        # Warnings
        if hacks:
            lines.append(f"\n  ⚠ REWARD HACKING: {', '.join(hacks)}")
            lines.append("    Fix: reduce weight of dominant component")

        # Specific diagnostics
        lines.append("\n  💡 Diagnostics:")
        if dom.get("pnl", 0) + dom.get("pnl_mtm", 0) < 0.25:
            lines.append(
                "    ✗ PnL signal < 25% — agent not learning from trades")
            lines.append("      → increase w_pnl or reward_scale")
        if dom.get("dd", 0) + dom.get("ddb", 0) > 0.50:
            lines.append(
                "    ✗ Drawdown penalty dominates — agent avoids all trades")
            lines.append(
                "      → reduce w_drawdown or raise max_drawdown_threshold")
        if spar.get("pnl", 0) < 0.05:
            lines.append("    ✗ PnL fires < 5% of steps — too sparse")
            lines.append("      → add MTM reward (pnl_mtm)")
        if dom.get("pnl", 0) > 0:
            lines.append("    ✓ PnL signal present")
        if dom.get("pnl_mtm", 0) > 0:
            lines.append("    ✓ MTM signal present (continuous feedback)")

        return "\n".join(lines)


# ══════════════════════════════════════════════════════
# CLASS 2: TrainingMonitor
# Purpose: track losses, detect divergence
# USE IN: after every trainer.update() call
# ══════════════════════════════════════════════════════

class TrainingMonitor:
    """
    Monitors PPO training health.

    Detects:
    - Policy loss explosion
    - Value loss explosion
    - KL divergence too high (policy changing too fast)
    - Entropy collapse (policy becomes deterministic)
    - Gradient norm explosion

    USAGE:
        monitor = TrainingMonitor()
        # after each update:
        monitor.log(update_metrics)
        issues = monitor.check()
        if issues:
            print(issues)
    """

    # Thresholds — raise an issue if exceeded
    THRESHOLDS = {
        "policy_loss":  (-5.0,  2.0),   # (min, max)
        "value_loss":   (0.0,   50.0),
        "approx_kl":    (0.0,   0.30),
        "entropy":      (0.05,  5.0),
        "grad_norm":    (0.0,   20.0),
        "clip_frac":    (0.0,   0.80),
    }

    def __init__(self, window: int = 50) -> None:
        self.window = window
        self.history: Dict[str, List[float]] = defaultdict(list)

    def log(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                self.history[k].append(float(v))

    def check(self) -> List[str]:
        """Return list of issue descriptions. Empty = healthy."""
        issues = []
        for metric, (lo, hi) in self.THRESHOLDS.items():
            vals = self.history.get(metric, [])
            if not vals:
                continue
            recent_mean = float(np.mean(vals[-self.window:]))
            if recent_mean < lo:
                issues.append(
                    f"  ⚠ {metric}={recent_mean:.4f} below min {lo}"
                )
            if recent_mean > hi:
                issues.append(
                    f"  ⚠ {metric}={recent_mean:.4f} above max {hi}"
                )

        # Entropy collapse check
        ent_vals = self.history.get("entropy", [])
        if len(ent_vals) > 20:
            early = float(np.mean(ent_vals[:10]))
            late = float(np.mean(ent_vals[-10:]))
            if early > 0.1 and late < early * 0.2:
                issues.append(
                    f"  ⚠ entropy collapsed: {early:.3f} → {late:.3f}"
                    f" (increase entropy_coef)"
                )

        # KL spike check
        kl_vals = self.history.get("approx_kl", [])
        if kl_vals and float(kl_vals[-1]) > 0.20:
            issues.append(
                f"  ⚠ KL spike: {kl_vals[-1]:.4f} (reduce learning_rate)"
            )

        return issues

    def summary(self, last_n: int = 10) -> str:
        lines = ["\n📈 TRAINING MONITOR"]
        for metric in ("policy_loss", "value_loss", "approx_kl",
                       "entropy", "grad_norm", "clip_frac", "lr"):
            vals = self.history.get(metric, [])
            if not vals:
                continue
            recent = vals[-last_n:]
            lines.append(
                f"  {metric:<15} mean={np.mean(recent):+.4f}"
                f"  last={recent[-1]:+.4f}"
            )
        issues = self.check()
        if issues:
            lines.append("\n  Issues:")
            lines.extend(issues)
        else:
            lines.append("\n  ✓ All metrics healthy")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════
# CLASS 3: EpisodeTracker
# Purpose: track episode outcomes, detect if agent is trading
# USE IN: end of each episode
# ══════════════════════════════════════════════════════

class EpisodeTracker:
    """
    Tracks episode-level statistics.

    Detects:
    - Agent not trading (0 trades = reward signal never fires)
    - Systematic losses (win rate < 30%)
    - Capital explosion or collapse

    USAGE:
        tracker = EpisodeTracker()
        # after episode ends (when res.done == True):
        tracker.log(env.episode_stats)
        print(tracker.summary())
    """

    def __init__(self, window: int = 50) -> None:
        self.window = window
        self.episodes: List[Dict[str, float]] = []

    def log_from_info(self, info: Dict[str, Any]) -> None:
        """Call with step result info dict when done=True."""
        if "episode_stats" not in info:
            return
        s = info["episode_stats"]
        self.episodes.append({
            "final_capital": float(s.final_capital),
            "total_trades":  int(s.total_trades),
            "win_rate":      float(s.win_rate),
            "max_drawdown":  float(s.max_drawdown),
            "total_reward":  float(s.total_reward),
        })

    def check(self) -> List[str]:
        issues = []
        recent = self.episodes[-self.window:]
        if not recent:
            return issues

        avg_trades = float(np.mean([e["total_trades"] for e in recent]))
        avg_wr = float(np.mean([e["win_rate"] for e in recent]))
        avg_cap = float(np.mean([e["final_capital"] for e in recent]))
        avg_dd = float(np.mean([e["max_drawdown"] for e in recent]))

        if avg_trades < 1:
            issues.append(
                "  ✗ Agent not trading (avg_trades < 1)"
                " → reward signal never fires from PnL"
                " → add MTM reward or reduce transaction costs"
            )
        if avg_trades > 0 and avg_wr < 0.30:
            issues.append(
                f"  ✗ Win rate {avg_wr:.1%} < 30%"
                " → agent entering at wrong times"
                " → check action parsing / direction signal"
            )
        if avg_cap < 70_000:
            issues.append(
                f"  ✗ Average capital {avg_cap:,.0f} < 70K"
                " → agent losing money systematically"
            )
        if avg_dd > 0.20:
            issues.append(
                f"  ✗ Average drawdown {avg_dd:.1%} > 20%"
                " → position sizing too aggressive"
            )
        return issues

    def summary(self) -> str:
        recent = self.episodes[-self.window:]
        if not recent:
            return "  No episodes recorded yet"

        lines = ["\n🎯 EPISODE TRACKER"]
        lines.append(f"  Episodes tracked: {len(self.episodes)}")
        lines.append(
            f"  Avg trades:    {np.mean([e['total_trades']  for e in recent]):.1f}"
        )
        lines.append(
            f"  Avg win rate:  {np.mean([e['win_rate']      for e in recent]):.1%}"
        )
        lines.append(
            f"  Avg capital:   {np.mean([e['final_capital'] for e in recent]):,.0f}"
        )
        lines.append(
            f"  Avg drawdown:  {np.mean([e['max_drawdown']  for e in recent]):.1%}"
        )
        issues = self.check()
        if issues:
            lines.append("\n  Issues found:")
            lines.extend(issues)
        else:
            lines.append("  ✓ Episode metrics look healthy")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════
# CLASS 4: CheckpointManager
# Purpose: proper save/restore with optimizer state
# USE IN: training loop (replace trainer.save/load)
# ══════════════════════════════════════════════════════

class CheckpointManager:
    """
    Saves and restores FULL trainer state.

    v3.6 bug: only saved net.state_dict(), not optimizer.
    When restored, optimizer momentum was cold → training instability.

    USAGE:
        ckpt = CheckpointManager("/tmp/drl_best.pt")
        # on improvement:
        ckpt.save(trainer)
        # on early stopping:
        ckpt.restore(trainer)
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.best_metric = -float("inf")

    def save(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        metric: float,
    ) -> bool:
        """Save if metric improved. Returns True if saved."""
        if metric <= self.best_metric:
            return False
        self.best_metric = metric
        import os
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        torch.save({
            "net":    net.state_dict(),
            "opt":    optimizer.state_dict(),
            "sched":  scheduler.state_dict() if scheduler else None,
            "step":   step,
            "metric": metric,
        }, self.path)
        return True

    def restore(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
    ) -> int:
        """Restore full state. Returns global step."""
        import os
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Checkpoint not found: {self.path}")
        ck = torch.load(self.path, map_location=device)
        net.load_state_dict(ck["net"])
        optimizer.load_state_dict(ck["opt"])
        if scheduler and ck.get("sched"):
            scheduler.load_state_dict(ck["sched"])
        print(f"  Restored checkpoint: metric={ck.get('metric', '?'):+.4f}"
              f"  step={ck.get('step', '?')}")
        return int(ck.get("step", 0))



class RewardDebugger:
    """Track reward components — detect hacking and sparse signals."""

    def __init__(self, window: int = 200) -> None:
        self.window = window
        self.comps:       Dict[str, List[float]] = defaultdict(list)
        self.totals:      List[float] = []
        # ← was missing, caused AttributeError
        self.ep_rewards:  List[float] = []

    def log(self, components: Dict[str, float], total: float) -> None:
        for k, v in components.items():
            self.comps[k].append(float(v))
        self.totals.append(float(total))

    def dominance(self) -> Dict[str, float]:
        if not self.comps:
            return {}
        result = {}
        for k, vs in self.comps.items():
            recent = vs[-self.window:]
            result[k] = float(np.mean(np.abs(recent))) if recent else 0.0
        total = sum(result.values()) + 1e-10
        return {k: v / total for k, v in result.items()}

    def check_hacking(self) -> List[str]:
        dom = self.dominance()
        return [f"{k}({v:.0%})" for k, v in dom.items() if v > 0.65]

    def sparsity(self) -> Dict[str, float]:
        result = {}
        for k, vs in self.comps.items():
            recent = vs[-self.window:]
            result[k] = float(np.mean([abs(v) > 1e-6 for v in recent]))
        return result

    def report(self) -> str:
        dom = self.dominance()
        spar = self.sparsity()
        hacks = self.check_hacking()
        lines = ["\n📊 REWARD ANALYSIS", "=" * 45]
        lines.append(f"  {'comp':<12} {'dominance':>10} {'sparsity':>10}")
        lines.append(f"  {'-'*12} {'-'*10} {'-'*10}")
        for k in sorted(dom, key=dom.get, reverse=True):
            bar = "█" * int(dom[k] * 15)
            lines.append(
                f"  {k:<12} {dom[k]:>9.1%}  {spar.get(k, 0):>9.1%}  {bar}"
            )
        if self.totals:
            recent = self.totals[-self.window:]
            lines.append(f"\n  Mean total: {np.mean(recent):+.4f}"
                         f"  Std: {np.std(recent):.4f}")
        if self.ep_rewards:
            lines.append(
                f"  Mean ep_R:  {np.mean(self.ep_rewards[-20:]):+.4f}")
        if hacks:
            lines.append(f"\n  ⚠ HACKING: {', '.join(hacks)}")
        lines.append("\n  💡 Diagnostics:")
        pnl_mass = dom.get("pnl", 0) + dom.get("pnl_mtm", 0)
        if pnl_mass < 0.20:
            lines.append("    ✗ PnL < 20% of reward — increase w_pnl")
        else:
            lines.append(f"    ✓ PnL signal = {pnl_mass:.0%}")
        if dom.get("dd", 0) + dom.get("ddb", 0) > 0.50:
            lines.append("    ✗ DD penalty > 50% — agent avoids trading")
        if spar.get("pnl", 1) < 0.05 and spar.get("pnl_mtm", 1) < 0.05:
            lines.append("    ✗ PnL fires < 5% — add MTM or reduce dead-zone")
        return "\n".join(lines)


class TrainingMonitor:
    """Detect training instability."""

    THRESHOLDS = {
        "policy_loss":  (-5.0, 2.0),
        "value_loss":   (0.0,  50.0),
        "approx_kl":    (0.0,  0.30),
        "entropy":      (0.05, 5.0),
        "grad_norm":    (0.0,  20.0),
        "clip_frac":    (0.0,  0.80),
    }

    def __init__(self, window: int = 50) -> None:
        self.window = window
        self.history: Dict[str, List[float]] = defaultdict(list)

    def log(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                self.history[k].append(float(v))

    def check(self) -> List[str]:
        issues = []
        for metric, (lo, hi) in self.THRESHOLDS.items():
            vals = self.history.get(metric, [])
            if not vals:
                continue
            mean = float(np.mean(vals[-self.window:]))
            if mean < lo:
                issues.append(f"⚠ {metric}={mean:.4f} below min {lo}")
            if mean > hi:
                issues.append(f"⚠ {metric}={mean:.4f} above max {hi}")
        # Entropy collapse
        ent = self.history.get("entropy", [])
        if len(ent) > 20:
            early = float(np.mean(ent[:10]))
            late = float(np.mean(ent[-10:]))
            if early > 0.1 and late < early * 0.15:
                issues.append(
                    f"⚠ entropy collapsed {early:.3f}→{late:.3f}"
                    f" (increase entropy_coef)")
        return issues

    def summary(self, last_n: int = 10) -> str:
        lines = ["\n📈 TRAINING MONITOR"]
        for metric in ("policy_loss", "value_loss", "approx_kl",
                       "entropy", "grad_norm", "clip_frac"):
            vals = self.history.get(metric, [])
            if not vals:
                continue
            recent = vals[-last_n:]
            lines.append(
                f"  {metric:<15} mean={np.mean(recent):+.4f}"
                f"  last={recent[-1]:+.4f}")
        issues = self.check()
        if issues:
            lines.append("\n  Issues:")
            for i in issues:
                lines.append(f"    {i}")
        else:
            lines.append("\n  ✓ All metrics healthy")
        return "\n".join(lines)


class EpisodeTracker:
    """Track episode outcomes and detect agent failure modes."""

    def __init__(self, window: int = 50) -> None:
        self.window = window
        self.episodes: List[Dict[str, float]] = []

    def log_from_info(self, info: Dict[str, Any]) -> None:
        if "episode_stats" not in info:
            return
        s = info["episode_stats"]
        self.episodes.append({
            "final_capital": float(getattr(s, "final_capital", 100_000)),
            "total_trades":  int(getattr(s,   "total_trades",  0)),
            "win_rate":      float(getattr(s,  "win_rate",      0)),
            "max_drawdown":  float(getattr(s,  "max_drawdown",  0)),
            "total_reward":  float(getattr(s,  "total_reward",  0)),
        })

    def check(self) -> List[str]:
        recent = self.episodes[-self.window:]
        if not recent:
            return []
        issues = []
        avg_trd = float(np.mean([e["total_trades"] for e in recent]))
        avg_wr = float(np.mean([e["win_rate"] for e in recent]))
        avg_cap = float(np.mean([e["final_capital"] for e in recent]))
        avg_dd = float(np.mean([e["max_drawdown"] for e in recent]))

        if avg_trd < 1:
            issues.append(
                "✗ 0 trades — dead-zone too wide or entropy too high")
        if avg_trd > 0 and avg_wr < 0.30:
            issues.append(
                f"✗ win_rate={avg_wr:.1%} < 30%")
        if avg_cap < 75_000:
            issues.append(
                f"✗ capital={avg_cap:,.0f} < 75K — losing money")
        if avg_dd > 0.25:
            issues.append(
                f"✗ drawdown={avg_dd:.1%} > 25% — size too large")
        return issues

    def summary(self) -> str:
        recent = self.episodes[-self.window:]
        if not recent:
            return "  No episodes yet"
        lines = ["\n🎯 EPISODE TRACKER",
                 f"  count:     {len(self.episodes)}",
                 f"  avg_trades: {np.mean([e['total_trades']  for e in recent]):.1f}",
                 f"  avg_wr:     {np.mean([e['win_rate']      for e in recent]):.1%}",
                 f"  avg_cap:    {np.mean([e['final_capital'] for e in recent]):,.0f}",
                 f"  avg_dd:     {np.mean([e['max_drawdown']  for e in recent]):.1%}"]
        for i in self.check():
            lines.append(f"  {i}")
        return "\n".join(lines)


class CheckpointManager:
    """Save/restore FULL trainer state (net + optimizer + scheduler)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.best_metric = -float("inf")

    def save(
        self,
        net:       nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step:      int,
        metric:    float,
    ) -> bool:
        if metric <= self.best_metric:
            return False
        self.best_metric = metric
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        torch.save({
            "net":    net.state_dict(),
            "opt":    optimizer.state_dict(),
            "sched":  scheduler.state_dict() if scheduler else None,
            "step":   step,
            "metric": metric,
        }, self.path)
        return True

    def restore(
        self,
        net:       nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device:    torch.device,
    ) -> int:
        if not os.path.exists(self.path):
            print(f"  No checkpoint at {self.path}")
            return 0
        ck = torch.load(self.path, map_location=device)
        net.load_state_dict(ck["net"])
        optimizer.load_state_dict(ck["opt"])
        if scheduler and ck.get("sched"):
            scheduler.load_state_dict(ck["sched"])
        print(f"  Restored: metric={ck.get('metric','?'):+.4f}"
              f"  step={ck.get('step','?')}")
        return int(ck.get("step", 0))
