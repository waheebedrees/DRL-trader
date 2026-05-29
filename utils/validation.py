
from __future__ import annotations

import numpy as np
from typing import List, Callable, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import copy

# Import from main package
import sys
sys.path.append('..')


@dataclass
class ValidationResult:
    """Results from a single validation fold"""
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    # In-sample metrics
    is_sharpe: float = 0.0
    is_return: float = 0.0
    is_max_dd: float = 0.0

    # Out-of-sample metrics
    oos_sharpe: float = 0.0
    oos_return: float = 0.0
    oos_max_dd: float = 0.0

    # Overfitting indicators
    overfitting_ratio: float = 0.0  # OOS/IS Sharpe ratio

    @property
    def is_overfit(self) -> bool:
        """Heuristic overfitting detection"""
        return self.overfitting_ratio < 0.3


@dataclass
class WalkForwardReport:
    """Complete walk-forward validation report"""
    results: List[ValidationResult] = field(default_factory=list)

    @property
    def avg_oos_sharpe(self) -> float:
        return np.mean([r.oos_sharpe for r in self.results]) if self.results else 0.0

    @property
    def avg_overfitting_ratio(self) -> float:
        return np.mean([r.overfitting_ratio for r in self.results]) if self.results else 0.0

    @property
    def overfit_folds(self) -> int:
        return sum(1 for r in self.results if r.is_overfit)

    def summary(self) -> str:
        lines = ["=" * 60, "WALK-FORWARD VALIDATION REPORT", "=" * 60]

        for r in self.results:
            status = "⚠ OVERFIT" if r.is_overfit else "✓ OK"
            lines.append(
                f"Fold {r.fold}: OOS Sharpe={r.oos_sharpe:.3f} "
                f"(IS={r.is_sharpe:.3f}, Ratio={r.overfitting_ratio:.3f}) {status}"
            )

        lines.append(f"\nAverage OOS Sharpe: {self.avg_oos_sharpe:.3f}")
        lines.append(f"Average OOS/IS Ratio: {self.avg_overfitting_ratio:.3f}")
        lines.append(
            f"Overfit Folds: {self.overfit_folds}/{len(self.results)}")

        if self.avg_overfitting_ratio > 0.7:
            lines.append(
                "✓ Good generalization - model transfers to unseen data")
        elif self.avg_overfitting_ratio > 0.4:
            lines.append("⚠ Moderate overfitting - monitor carefully")
        else:
            lines.append("✗ Severe overfitting - model doesn't generalize")

        lines.append("=" * 60)
        return "\n".join(lines)


class WalkForwardValidator:
    """
    Walk-forward (expanding window) validation for time series.
    
    Usage:
        validator = WalkForwardValidator(
            train_size=1000,
            test_size=200,
            n_splits=5
        )
        
        # Define factory functions
        def create_env(bars):
            return SingleAssetEnv(bars, config, device)
        
        def create_trainer(net):
            return PPOTrainer(net, ppo_config, device)
        
        # Run validation
        report = validator.run(
            all_bars=bars,
            net_factory=lambda: ACNet(...),
            trainer_factory=create_trainer,
            env_factory=create_env,
            train_fn=train_model,  # Your training function
            eval_fn=evaluate_model,  # Your evaluation function
        )
        
        print(report.summary())
    """

    def __init__(self, train_size: int, test_size: int,
                 n_splits: int = 5, min_train_size: int = 500):
        self.train_size = train_size
        self.test_size = test_size
        self.n_splits = n_splits
        self.min_train_size = min_train_size

    def run(self, all_bars: List,
            net_factory: Callable,
            trainer_factory: Callable,
            env_factory: Callable,
            train_fn: Callable,
            eval_fn: Callable,
            verbose: bool = True) -> WalkForwardReport:
        """
        Run walk-forward validation.
        
        Args:
            all_bars: Complete bar data
            net_factory: Function that creates a new network
            trainer_factory: Function that creates a trainer given a network
            env_factory: Function that creates an environment given bars
            train_fn: Function(net, trainer, train_env) -> metrics
            eval_fn: Function(net, trainer, eval_env) -> metrics
            verbose: Print progress
        
        Returns:
            WalkForwardReport with per-fold results
        """
        report = WalkForwardReport()
        total_bars = len(all_bars)

        for fold in range(self.n_splits):
            # Calculate window boundaries
            train_start = 0
            train_end = min(self.train_size + fold *
                            self.test_size, total_bars)
            test_start = train_end
            test_end = min(test_start + self.test_size, total_bars)

            if test_end - test_start < self.min_train_size:
                if verbose:
                    print(
                        f"Fold {fold}: Not enough data for test window. Stopping.")
                break

            train_bars = all_bars[train_start:train_end]
            test_bars = all_bars[test_start:test_end]

            if verbose:
                print(f"\n{'='*50}")
                print(f"Fold {fold + 1}/{self.n_splits}")
                print(
                    f"Train: bars {train_start}-{train_end} ({len(train_bars)} bars)")
                print(
                    f"Test:  bars {test_start}-{test_end} ({len(test_bars)} bars)")

            # Create fresh model for each fold
            net = net_factory()
            trainer = trainer_factory(net)

            # Create environments
            train_env = env_factory(train_bars)
            test_env = env_factory(test_bars)

            # Train
            if verbose:
                print("Training...")
            train_fn(net, trainer, train_env)

            # Evaluate in-sample
            if verbose:
                print("Evaluating in-sample...")
            is_metrics = eval_fn(net, trainer, train_env)

            # Evaluate out-of-sample
            if verbose:
                print("Evaluating out-of-sample...")
            oos_metrics = eval_fn(net, trainer, test_env)

            # Calculate metrics
            is_sharpe = self._extract_sharpe(is_metrics)
            oos_sharpe = self._extract_sharpe(oos_metrics)

            result = ValidationResult(
                fold=fold + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                is_sharpe=is_sharpe,
                oos_sharpe=oos_sharpe,
                is_return=self._extract_return(is_metrics),
                oos_return=self._extract_return(oos_metrics),
                overfitting_ratio=oos_sharpe / max(is_sharpe, 1e-8),
            )

            report.results.append(result)

            if verbose:
                print(f"  IS Sharpe:  {is_sharpe:.3f}")
                print(f"  OOS Sharpe: {oos_sharpe:.3f}")
                print(f"  OOS/IS:     {result.overfitting_ratio:.3f}")

        return report

    def _extract_sharpe(self, metrics: Dict) -> float:
        """Extract Sharpe ratio from metrics dict"""
        # Try different keys that might contain Sharpe
        for key in ['sharpe_ratio', 'sharpe', 'mean_sharpe']:
            if key in metrics:
                return float(metrics[key])

        # Approximate from mean/std reward
        if 'mean_reward' in metrics and 'std_reward' in metrics:
            return metrics['mean_reward'] / max(metrics['std_reward'], 1e-8)

        # Approximate from returns
        if 'total_return' in metrics and 'max_drawdown' in metrics:
            # Rough approximation
            return metrics['total_return'] / max(metrics['max_drawdown'], 0.01) * 0.5

        return 0.0

    def _extract_return(self, metrics: Dict) -> float:
        """Extract total return from metrics dict"""
        for key in ['total_return', 'mean_return', 'return']:
            if key in metrics:
                return float(metrics[key])
        return 0.0


class CrossValidator:
    """
    Purged cross-validation for time series (not standard K-fold).
    
    This prevents data leakage by:
    1. Maintaining temporal order
    2. Purging overlapping data
    3. Embargo periods between train/test
    """

    def __init__(self, n_splits: int = 5, purge_size: int = 50,
                 embargo_size: int = 10):
        self.n_splits = n_splits
        self.purge_size = purge_size
        self.embargo_size = embargo_size

    def split(self, n_bars: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged train/test splits.
        
        Returns:
            List of (train_indices, test_indices) tuples
        """
        splits = []

        # Calculate test window size
        test_size = n_bars // self.n_splits

        for i in range(self.n_splits):
            test_start = i * test_size
            test_end = min(test_start + test_size, n_bars)

            # Train is everything before test_start (minus embargo)
            train_end = max(0, test_start - self.embargo_size)

            # Purge overlapping data near boundary
            purge_end = min(train_end + self.purge_size, test_start)

            if train_end > self.purge_size:
                train_indices = np.arange(0, train_end - self.purge_size)
            else:
                train_indices = np.arange(0, train_end)

            test_indices = np.arange(test_start, test_end)

            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))

        return splits
