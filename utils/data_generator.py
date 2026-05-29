"""
Realistic Market Data Generator
===============================
Generate synthetic market data with realistic statistical properties
for training trading agents.

Features:
- Regime switching (trending, mean-reverting, volatile)
- Volatility clustering (GARCH-like)
- Fat-tailed returns
- Price-volume correlation
- Microstructure noise (bid-ask bounce)
- Multiple asset correlation (optional)
"""

from __future__ import annotations
from state import Bar, TimeFrame,MarketRegime
from config import MarketConfig


import numpy as np
from typing import List, Optional, Tuple, Dict

# Import from main package (adjust import path as needed)
import sys
sys.path.append('..')



class RealisticMarketSimulator:
    """
    Generate realistic market data with proper statistical properties.
    
    Usage:
        sim = RealisticMarketSimulator(seed=42)
        bars = sim.generate(n_bars=10000, symbol="BTC/USDT")
    """

    def __init__(self, config: Optional[MarketConfig] = None, seed: int = 42):
        self.config = config or MarketConfig()
        self.rng = np.random.default_rng(seed)

        # State variables
        self.price: float = self.config.initial_price
        self.volatility: float = self.config.volatility_base
        self.regime: MarketRegime = MarketRegime.SIDEWAYS
        self.returns_history: List[float] = []
        self.volume_history: List[float] = []

        # For reproducible regime changes
        self.regime_duration: int = 0
        self.regime_timer: int = 0

    def _next_regime(self) -> MarketRegime:
        """Transition to next regime based on transition matrix"""
        if self.regime_timer > 0:
            self.regime_timer -= 1
            return self.regime

        # Get transition probabilities for current regime
        current_idx = abs(self.regime.value)  # Map -2,-1,0,1,2 to 0,1,2,3,4
        probs = self.config.regime_transitions[current_idx]

        # Sample new regime
        new_idx = self.rng.choice(5, p=probs)
        new_regime = MarketRegime(-2 + new_idx)  # Map back to -2,-1,0,1,2

        # Set duration
        self.regime_duration = self.rng.integers(50, 200)
        self.regime_timer = self.regime_duration

        return new_regime

    def _get_regime_drift(self) -> float:
        """Get expected return for current regime"""
        drift_map = {
            MarketRegime.CRASH: -0.003,
            MarketRegime.BEAR: -0.0005,
            MarketRegime.SIDEWAYS: 0.0001,
            MarketRegime.BULL: 0.001,
            MarketRegime.BUBBLE: 0.003,
        }
        return drift_map[self.regime]

    def _update_volatility(self, current_return: float) -> None:
        """Update volatility with GARCH-like dynamics"""
        shock = abs(current_return) if self.returns_history else 0

        if len(self.returns_history) >= 20:
            recent_vol = np.std(self.returns_history[-20:])
        else:
            recent_vol = self.config.volatility_base

        self.volatility = np.clip(
            0.005 +
            self.config.volatility_persistence * self.volatility +
            0.1 * recent_vol +
            0.05 * shock,
            0.005, 0.15
        )

    def _generate_return(self, drift: float) -> float:
        """Generate return with fat tails"""
        # Student's t approximation via normal mixture
        if self.rng.random() < self.config.fat_tail_prob:
            return self.rng.normal(drift, self.volatility * 3)
        else:
            return self.rng.normal(drift, self.volatility)

    def _generate_volume(self, return_val: float, regime_change: bool) -> float:
        """Generate volume correlated with price moves"""
        multiplier = 1.0

        # Higher volume on large moves
        if abs(return_val) > self.volatility * 1.5:
            multiplier += abs(return_val) / self.volatility * 0.5

        # Higher volume on down moves (fear)
        if return_val < -self.volatility:
            multiplier += 0.3

        # Higher volume on regime changes
        if regime_change:
            multiplier += 0.5

        # Higher volume in extreme regimes
        if self.regime in [MarketRegime.CRASH, MarketRegime.BUBBLE]:
            multiplier += 0.2

        return float(self.rng.lognormal(
            np.log(self.config.base_volume) + np.log(max(multiplier, 0.1)),
            0.3
        ))

    def generate(self, n_bars: int, symbol: str = "SYNTH/USD",
                 timeframe: TimeFrame = TimeFrame.H1,
                 start_timestamp: float = 1_700_000_000) -> List[Bar]:
        """
        Generate n_bars of realistic market data.
        
        Args:
            n_bars: Number of bars to generate
            symbol: Trading symbol name
            timeframe: Bar timeframe
            start_timestamp: Unix timestamp for first bar
        
        Returns:
            List of Bar objects
        """
        bars: List[Bar] = []

        for i in range(n_bars):
            # Update regime
            new_regime = self._next_regime()
            regime_changed = (new_regime != self.regime)
            self.regime = new_regime

            # Calculate drift with momentum/mean-reversion effects
            drift = self._get_regime_drift()

            if len(self.returns_history) >= 10:
                momentum_10 = np.mean(self.returns_history[-10:])

                # Momentum works in trending regimes
                if self.regime in [MarketRegime.BULL, MarketRegime.BUBBLE]:
                    drift += momentum_10 * 0.1
                elif self.regime in [MarketRegime.BEAR, MarketRegime.CRASH]:
                    drift += momentum_10 * 0.1  # Trend following in bear too

                # Mean reversion in sideways
                if self.regime == MarketRegime.SIDEWAYS and len(self.returns_history) >= 50:
                    long_term = np.mean(self.returns_history[-50:])
                    drift -= long_term * 0.02

            # Generate return
            ret = self._generate_return(drift)
            ret = np.clip(ret, -self.config.max_return, self.config.max_return)

            # Update price
            self.price *= np.exp(ret)
            self.price = np.clip(
                self.price, self.config.min_price, self.config.max_price)

            # Update volatility
            self._update_volatility(ret)

            # Generate OHLC with realistic patterns
            intraday_range = self.price * \
                self.volatility * self.rng.uniform(0.5, 2.0)

            if ret > self.volatility * 0.5:  # Strong up
                hi = self.price + intraday_range * self.rng.uniform(0.3, 1.0)
                lo = self.price - intraday_range * self.rng.uniform(0.1, 0.5)
                op = self.price - intraday_range * self.rng.uniform(0.1, 0.4)
            elif ret < -self.volatility * 0.5:  # Strong down
                hi = self.price + intraday_range * self.rng.uniform(0.1, 0.5)
                lo = self.price - intraday_range * self.rng.uniform(0.3, 1.0)
                op = self.price + intraday_range * self.rng.uniform(0.1, 0.4)
            else:  # Sideways
                hi = self.price + intraday_range * self.rng.uniform(0.2, 0.8)
                lo = self.price - intraday_range * self.rng.uniform(0.2, 0.8)
                op = self.price + intraday_range * self.rng.uniform(-0.2, 0.2)

            # Apply microstructure noise (bid-ask bounce)
            noise = self.config.microstructure_noise * self.rng.uniform(-1, 1)

            close = self.price
            open_price = max(op + noise * self.price, self.config.min_price)
            high = max(hi + noise * self.price, open_price, close)
            low = min(lo + noise * self.price, open_price, close)

            # Generate volume
            volume = self._generate_volume(ret, regime_changed)
            volume = max(volume, 100.0)

            # Create bar
            timestamp = start_timestamp + i * 3600
            bar = Bar(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                open=float(open_price),
                high=float(high),
                low=float(low),
                close=float(close),
                volume=float(volume),
            )

            bars.append(bar)

            # Update history
            self.returns_history.append(ret)
            if len(self.returns_history) > 200:
                self.returns_history.pop(0)

            self.volume_history.append(volume)
            if len(self.volume_history) > 200:
                self.volume_history.pop(0)

        return bars

    def reset(self) -> None:
        """Reset simulator state"""
        self.price = self.config.initial_price
        self.volatility = self.config.volatility_base
        self.regime = MarketRegime.SIDEWAYS
        self.returns_history.clear()
        self.volume_history.clear()
        self.regime_duration = 0
        self.regime_timer = 0


class DataGenerator:
    """
    High-level data generator with train/val/test splitting.
    
    Usage:
        gen = DataGenerator(seed=42)
        
        # Generate 10k bars split into train/val/test
        train, val, test = gen.generate_split(
            total_bars=10000,
            train_ratio=0.7,
            val_ratio=0.15,
        )
    """

    def __init__(self, seed: int = 42):
        self.simulator = RealisticMarketSimulator(seed=seed)
        self.seed = seed

    def generate(self, n_bars: int, symbol: str = "SYNTH/USD",
                 timeframe: TimeFrame = TimeFrame.H1) -> List[Bar]:
        """Generate n_bars of data"""
        self.simulator.reset()
        return self.simulator.generate(n_bars, symbol, timeframe)

    def generate_split(self, total_bars: int = 10000,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       test_ratio: float = 0.15,
                       symbol: str = "SYNTH/USD") -> Tuple[List[Bar], List[Bar], List[Bar]]:
        """
        Generate data and split into train/validation/test sets.
        
        Ensures no time-series leakage by maintaining temporal order.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01

        all_bars = self.generate(total_bars, symbol)

        train_end = int(total_bars * train_ratio)
        val_end = train_end + int(total_bars * val_ratio)

        train_bars = all_bars[:train_end]
        val_bars = all_bars[train_end:val_end]
        test_bars = all_bars[val_end:]

        return train_bars, val_bars, test_bars

    def generate_multiple_regimes(self, bars_per_regime: int = 2000,
                                  regimes: Optional[List[MarketRegime]] = None,
                                  symbol: str = "SYNTH/USD") -> List[Bar]:
        """
        Generate data with forced regime changes for testing.
        Each regime gets bars_per_regime bars.
        """
        if regimes is None:
            regimes = list(MarketRegime)

        all_bars = []

        for regime in regimes:
            # Force regime by setting transition matrix
            original_transitions = self.simulator.config.regime_transitions.copy()

            # Make transition matrix stay in this regime
            idx = abs(regime.value)
            stay_matrix = np.zeros((5, 5))
            stay_matrix[idx, idx] = 0.98  # 98% chance stay
            stay_matrix[idx, (idx + 1) % 5] = 0.02  # 2% chance next

            self.simulator.config.regime_transitions = stay_matrix
            self.simulator.regime = regime

            # Generate bars for this regime
            regime_bars = self.simulator.generate(bars_per_regime,
                                                  f"{symbol}_{regime.name}")
            all_bars.extend(regime_bars)

            # Restore transitions
            self.simulator.config.regime_transitions = original_transitions
            self.simulator.reset()

        return all_bars
