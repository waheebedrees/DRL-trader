"""
Training Diagnostics & Monitoring
=================================
Tools to understand what your agent is learning (or not learning).

Key features:
- Track action distributions to detect policy collapse
- Monitor reward components to find reward hacking
- Analyze value function accuracy
- Detect overfitting indicators
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class DiagnosticReport:
    """Comprehensive training diagnostic report"""
    policy_health: str = "unknown"
    overfitting_risk: str = "unknown"
    reward_dominance: Dict[str, float] = field(default_factory=dict)
    action_statistics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["=" * 60, "DIAGNOSTIC REPORT", "=" * 60]
        lines.append(f"Policy Health: {self.policy_health}")
        lines.append(f"Overfitting Risk: {self.overfitting_risk}")

        if self.reward_dominance:
            lines.append("\nReward Component Dominance:")
            for k, v in sorted(self.reward_dominance.items(),
                               key=lambda x: abs(x[1]), reverse=True):
                lines.append(f"  {k:20s}: {v:+.4f}")

        if self.action_statistics:
            lines.append("\nAction Statistics:")
            for k, v in self.action_statistics.items():
                lines.append(f"  {k:20s}: {v:.4f}")

        if self.recommendations:
            lines.append("\n📋 Recommendations:")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"  {i}. {rec}")

        lines.append("=" * 60)
        return "\n".join(lines)


class RewardAnalyzer:
    """
    Track and analyze reward components during training.
    
    Usage:
        analyzer = RewardAnalyzer()
        
        # During training loop:
        for step in range(n_steps):
            reward, components = env.step(action)
            analyzer.update(components)
            
            if step % 100 == 0:
                report = analyzer.get_report()
                print(report)
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.episode_rewards: List[float] = []
        self._current_episode_reward = 0.0
        self._step_count = 0

    def update(self, components: Dict[str, float]) -> None:
        """Add reward components from a single step"""
        for key, value in components.items():
            self.history[key].append(float(value))
        self._step_count += 1

    def update_episode(self, total_reward: float) -> None:
        """Called at end of episode"""
        self.episode_rewards.append(total_reward)

    def get_dominance(self) -> Dict[str, float]:
        """Calculate which reward components dominate"""
        if not self.history:
            return {}

        dominance = {}
        for key, values in self.history.items():
            if len(values) > 10:
                # Use recent window
                recent = values[-self.window_size:]
                dominance[key] = np.mean(np.abs(recent))

        # Normalize
        total = sum(dominance.values())
        if total > 0:
            dominance = {k: v/total for k, v in dominance.items()}

        return dominance

    def get_trend(self, key: str) -> str:
        """Detect trend in a reward component"""
        if key not in self.history or len(self.history[key]) < 50:
            return "insufficient_data"

        values = self.history[key][-self.window_size:]
        x = np.arange(len(values))

        # Simple linear regression slope
        slope = np.polyfit(x, values, 1)[0]

        if slope > 0.001:
            return "increasing"
        elif slope < -0.001:
            return "decreasing"
        else:
            return "stable"

    def get_report(self) -> DiagnosticReport:
        """Generate comprehensive diagnostic report"""
        report = DiagnosticReport()
        report.reward_dominance = self.get_dominance()

        # Check for reward hacking (one component dominates)
        if report.reward_dominance:
            max_component = max(report.reward_dominance,
                                key=lambda k: report.reward_dominance[k])
            max_value = report.reward_dominance[max_component]

            if max_value > 0.7:
                report.policy_health = "reward_hacking"
                report.recommendations.append(
                    f"'{max_component}' dominates {max_value:.0%} of reward. "
                    f"Agent may be exploiting this component. Consider reducing its weight."
                )
            elif max_value > 0.5:
                report.policy_health = "warning"
                report.recommendations.append(
                    f"'{max_component}' is {max_value:.0%} of reward. Monitor closely."
                )
            else:
                report.policy_health = "healthy"

        # Check episode rewards trend
        if len(self.episode_rewards) > 5:
            recent_rewards = self.episode_rewards[-10:]
            if len(recent_rewards) > 1:
                reward_slope = np.polyfit(range(len(recent_rewards)),
                                          recent_rewards, 1)[0]
                if reward_slope < -0.1:
                    report.recommendations.append(
                        "Episode rewards are declining. Consider reducing learning rate."
                    )

        return report

    def reset(self) -> None:
        """Reset all history"""
        self.history.clear()
        self.episode_rewards.clear()
        self._current_episode_reward = 0.0
        self._step_count = 0


class PolicyMonitor:
    """
    Monitor policy behavior to detect:
    - Mode collapse (always same action)
    - Entropy death (no exploration)
    - Action saturation (always at extremes)
    
    Usage:
        monitor = PolicyMonitor(action_dim=3)
        
        # During training:
        actions = model.get_action(state)
        stats = monitor.update(actions)
        print(stats)
    """

    def __init__(self, action_dim: int, window_size: int = 500):
        self.action_dim = action_dim
        self.window_size = window_size
        self.action_history: List[np.ndarray] = []
        self.entropy_history: List[float] = []

    def update(self, action: np.ndarray) -> Dict[str, float]:
        """Record action and return statistics"""
        action = np.asarray(action).flatten()
        self.action_history.append(action)

        if len(self.action_history) > self.window_size * 2:
            self.action_history = self.action_history[-self.window_size:]

        return self.get_stats()

    def update_entropy(self, entropy: float) -> None:
        """Record policy entropy"""
        self.entropy_history.append(entropy)
        if len(self.entropy_history) > self.window_size:
            self.entropy_history = self.entropy_history[-self.window_size:]

    def get_stats(self) -> Dict[str, float]:
        """Calculate policy statistics"""
        if not self.action_history:
            return {}

        actions = np.array(self.action_history)
        recent = actions[-self.window_size:] if len(
            actions) > self.window_size else actions

        stats = {
            "action_mean": float(np.mean(recent, axis=0)[0]),
            "action_std": float(np.mean(np.std(recent, axis=0))),
            "action_range": float(np.max(recent) - np.min(recent)),
            "saturation_ratio": float(np.mean(np.abs(recent) > 0.95)),
            "zero_action_ratio": float(np.mean(np.abs(recent[:, 0]) < 0.02)),
        }

        if self.entropy_history:
            recent_entropy = self.entropy_history[-self.window_size:]
            stats["policy_entropy"] = float(np.mean(recent_entropy))

            # Check for entropy death
            if stats["policy_entropy"] < 0.01:
                stats["warning"] = "entropy_death"

        return stats

    def is_collapsed(self) -> bool:
        """Check if policy has collapsed"""
        stats = self.get_stats()
        if not stats:
            return False

        # Policy collapse indicators:
        # 1. Very low action variance
        # 2. High saturation (always at extremes)
        # 3. Very low entropy
        is_low_variance = stats.get("action_std", 1.0) < 0.05
        is_saturated = stats.get("saturation_ratio", 0) > 0.8
        is_low_entropy = stats.get("policy_entropy", 1.0) < 0.001

        return is_low_variance or is_saturated or is_low_entropy

    def reset(self) -> None:
        """Clear history"""
        self.action_history.clear()
        self.entropy_history.clear()


class Diagnostics:
    """
    Main diagnostics class combining all monitoring tools.
    
    Usage:
        diag = Diagnostics(action_dim=3)
        
        for epoch in range(n_epochs):
            # Collect rollout
            for step in range(n_steps):
                action = model.get_action(state)
                reward, components = env.step(action)
                
                diag.update_step(action, components)
            
            # End of epoch
            diag.update_epoch(model, trainer)
            
            if epoch % 10 == 0:
                print(diag.get_full_report())
    """

    def __init__(self, action_dim: int, window_size: int = 500):
        self.reward_analyzer = RewardAnalyzer(window_size)
        self.policy_monitor = PolicyMonitor(action_dim, window_size)
        self.epoch_metrics: List[Dict] = []

    def update_step(self, action: np.ndarray, reward_components: Dict[str, float]):
        """Update after each environment step"""
        self.policy_monitor.update(action)
        self.reward_analyzer.update(reward_components)

    def update_epoch(self, model=None, trainer=None, metrics: Optional[Dict] = None):
        """Update after each training epoch"""
        if metrics:
            self.epoch_metrics.append(metrics)

    def get_full_report(self) -> DiagnosticReport:
        """Generate comprehensive report"""
        reward_report = self.reward_analyzer.get_report()
        policy_stats = self.policy_monitor.get_stats()

        report = DiagnosticReport(
            policy_health=reward_report.policy_health,
            reward_dominance=reward_report.reward_dominance,
            action_statistics=policy_stats,
            recommendations=reward_report.recommendations,
        )

        # Add policy-specific recommendations
        if self.policy_monitor.is_collapsed():
            report.recommendations.append(
                "⚠ Policy appears collapsed (low variance/entropy). "
                "Increase entropy coefficient or reduce learning rate."
            )

        if policy_stats.get("zero_action_ratio", 0) > 0.5:
            report.recommendations.append(
                "Agent choosing flat/no-action >50% of time. "
                "Check if inaction reward is too high."
            )

        # Overfitting check
        if len(self.epoch_metrics) > 20:
            recent_losses = [m.get("total_loss", 0) for m in self.epoch_metrics[-10:]
                             if "total_loss" in m]
            earlier_losses = [m.get("total_loss", 0) for m in self.epoch_metrics[-20:-10]
                              if "total_loss" in m]

            if recent_losses and earlier_losses:
                if np.mean(recent_losses) < np.mean(earlier_losses) * 0.1:
                    report.overfitting_risk = "high"
                    report.recommendations.append(
                        "Loss decreasing rapidly - may be memorizing patterns. "
                        "Increase dropout or reduce model size."
                    )

        return report

    def reset(self) -> None:
        """Reset all monitoring"""
        self.reward_analyzer.reset()
        self.policy_monitor.reset()
        self.epoch_metrics.clear()
