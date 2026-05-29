# data_gen.py  — CREATE THIS FILE
"""
Synthetic market data generators.
Isolated so they can be tested and replaced independently.
"""

from __future__ import annotations
from typing import List, Tuple

import math
from typing import List

import numpy as np

from state import Bar, TimeFrame


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


def make_bars_v3(
    n: int = 8000,
    seed: int = 42,
    symbol: str = "BTC/USDT",
) -> List[Bar]:
    """
    GARCH(1,1) + 5-regime switching + fat tails.

    Regimes: 0=crash  1=bear  2=sideways  3=bull  4=bubble

    Why this is better than v1/v2:
    - True volatility clustering (GARCH) — not just random vol
    - Realistic regime persistence via transition matrix
    - Volume correlated with fear/greed (not just random)
    - All OHLC values mathematically consistent
    - Fat tail events (2% probability of 3x shock)
    """
    rng = np.random.default_rng(seed)

    # ── initial state ─────────────────────────────────────────────
    price = 100.0
    vol = 0.015       # conditional volatility
    regime = 2           # start sideways
    reg_timer = int(rng.integers(50, 150))
    ret_hist: List[float] = []

    # ── GARCH(1,1) parameters ─────────────────────────────────────
    OMEGA = 0.000002
    ALPHA_G = 0.08
    BETA_G = 0.90

    # ── regime properties ─────────────────────────────────────────
    REG_DRIFT = {0: -0.004, 1: -0.001, 2: 0.0001, 3: 0.001,  4: 0.003}
    REG_VOL = {0: 2.5,   1: 1.5,    2: 1.0,    3: 1.2,    4: 2.0}

    # ── 5×5 transition matrix ─────────────────────────────────────
    T_MAT = np.array([
        [0.60, 0.25, 0.10, 0.04, 0.01],  # from crash
        [0.08, 0.55, 0.25, 0.10, 0.02],  # from bear
        [0.03, 0.12, 0.60, 0.20, 0.05],  # from sideways
        [0.01, 0.08, 0.25, 0.55, 0.11],  # from bull
        [0.15, 0.20, 0.20, 0.20, 0.25],  # from bubble
    ])

    bars: List[Bar] = []
    base_ts = 1_700_000_000

    for i in range(n):
        # ── regime switching ──────────────────────────────────────
        if reg_timer <= 0:
            regime = int(rng.choice(5, p=T_MAT[regime]))
            reg_timer = int(rng.integers(40, 200))
        reg_timer -= 1

        # ── GARCH volatility update ───────────────────────────────
        last_ret = ret_hist[-1] if ret_hist else 0.0
        vol = math.sqrt(
            max(OMEGA + ALPHA_G * last_ret ** 2 + BETA_G * vol ** 2, 1e-8)
        )
        eff_vol = float(np.clip(vol * REG_VOL[regime], 0.002, 0.15))

        # ── return generation ─────────────────────────────────────
        drift = REG_DRIFT[regime]

        # momentum in trending regimes
        if len(ret_hist) >= 10 and regime in (3, 4):
            drift += float(np.mean(ret_hist[-10:])) * 0.15

        # mean-reversion in sideways
        if len(ret_hist) >= 30 and regime == 2:
            drift -= float(np.mean(ret_hist[-30:])) * 0.05

        # fat tail: 2% chance
        if rng.random() < 0.02:
            ret = float(rng.normal(drift, eff_vol * 3.0))
        else:
            ret = float(rng.normal(drift, eff_vol))

        ret = float(np.clip(ret, -0.12, 0.12))

        # ── price update ──────────────────────────────────────────
        price = float(np.clip(price * math.exp(ret), 1.0, 1_000_000.0))
        ret_hist.append(ret)
        if len(ret_hist) > 200:
            ret_hist.pop(0)

        # ── OHLCV construction ────────────────────────────────────
        rng_pct = float(np.clip(abs(rng.normal(0, eff_vol * 1.5)), 0, 0.08))
        spread = price * float(rng.uniform(0.0002, 0.001))

        # realistic open (overnight gap)
        gap = float(rng.normal(0, eff_vol * 0.3))
        open_p = float(np.clip(price * math.exp(gap), 1.0, 1_000_000.0))

        # high/low consistent with bar direction
        if ret >= 0:
            high = price + price * rng_pct * float(rng.uniform(0.4, 1.0))
            low = open_p - price * rng_pct * float(rng.uniform(0.1, 0.5))
        else:
            high = open_p + price * rng_pct * float(rng.uniform(0.1, 0.5))
            low = price - price * rng_pct * float(rng.uniform(0.4, 1.0))

        # guarantee OHLC consistency
        high = max(high, open_p, price) + spread
        low = min(low,  open_p, price) - spread
        low = max(low, 1e-3)

        # volume: fear = high volume on down moves
        fear_mult = 1.5 if ret < -eff_vol else 1.0
        vol_mult = 1.0 + abs(ret) / max(eff_vol, 1e-8) * 0.4
        if regime in (0, 4):
            vol_mult *= 1.5
        volume = float(rng.lognormal(
            math.log(max(1_000_000 * vol_mult * fear_mult, 1.0)), 0.35
        ))

        bars.append(Bar(
            symbol=symbol,
            timeframe=TimeFrame.H1,
            timestamp=base_ts + i * 3600,
            open=float(open_p),
            high=float(high),
            low=float(low),
            close=float(price),
            volume=float(volume),
        ))

    return bars


def augment_bars(bars: List[Bar], noise_std: float = 0.002) -> List[Bar]:
    """
    Multiplicative noise on OHLCV.
    Prevents the network memorizing exact price levels.
    Call this every training iteration with fresh noise.
    """
    if noise_std <= 0:
        return list(bars)

    rng = np.random.default_rng()   # fresh seed each call
    out: List[Bar] = []

    for b in bars:
        nf = 1.0 + float(rng.normal(0, noise_std))
        out.append(Bar(
            symbol=b.symbol,
            timeframe=b.timeframe,
            timestamp=b.timestamp,
            open=max(float(b.open * nf), 1e-3),
            high=max(float(b.high * nf), 1e-3),
            low=max(float(b.low * nf), 1e-3),
            close=max(float(b.close * nf), 1e-3),
            volume=max(float(b.volume),   1.0),
        ))

    return out


def split_bars(
    bars: List[Bar],
    train_pct: float = 0.60,
    val_pct:   float = 0.20,
) -> tuple:
    """
    Walk-forward split — never shuffles.
    Returns (train, val, test).
    """
    n = len(bars)
    t_end = int(n * train_pct)
    v_end = int(n * (train_pct + val_pct))
    return bars[:t_end], bars[t_end:v_end], bars[v_end:]


# data_gen.py


def make_bars(
    n: int = 1200,
    seed: int = 42,
    symbol: str = "BTC/USDT",
) -> List[Bar]:
    """
    Simple v1 bars for trader.py smoke tests.
    Kept as-is so trader.py tests still pass.
    """
    rng = np.random.default_rng(seed)
    price = 50_000.0
    base = 1_700_000_000
    bars: List[Bar] = []

    for i in range(n):
        ret = float(rng.normal(0.0002, 0.012))
        ret = float(np.clip(ret, -0.10, 0.10))
        price = float(np.clip(price * math.exp(ret), 100.0, 1_000_000.0))
        rng_p = abs(float(rng.normal(0, 0.004)))
        hi = price * (1 + rng_p)
        lo = max(price * (1 - rng_p), 0.01)
        op = float(np.clip(
            price * rng.uniform(max(0.99, lo / price), min(1.01, hi / price)),
            0.01, 1_000_000.0,
        ))
        vol = float(rng.lognormal(9.5, 0.4))
        bars.append(Bar("BTC/USDT", TimeFrame.H1, base + i * 3600,
                        op, hi, lo, price, max(vol, 1.0)))
    return bars


def make_bars_v3(
    n: int = 10000,
    seed: int = 42,
    symbol: str = "BTC/USDT",
) -> List[Bar]:
    """
    GARCH(1,1) + 5-regime switching.
    Stronger regime signals so agent can learn something.
    """
    rng = np.random.default_rng(seed)
    price = 100.0
    vol = 0.012
    regime = 2
    reg_timer = int(rng.integers(60, 120))
    ret_hist: List[float] = []

    OMEGA, ALPHA_G, BETA_G = 0.000005, 0.10, 0.85

    REG_DRIFT = {0: -0.005, 1: -0.0015, 2: 0.0001, 3: 0.0015, 4: 0.004}
    REG_VOL = {0: 2.0,    1: 1.4,     2: 0.8,    3: 1.1,    4: 1.8}
    REG_DUR = {0: (40, 80), 1: (60, 120), 2: (80, 160),
               3: (60, 120), 4: (40, 80)}

    T_MAT = np.array([
        [0.50, 0.30, 0.15, 0.04, 0.01],
        [0.06, 0.50, 0.30, 0.12, 0.02],
        [0.02, 0.10, 0.65, 0.20, 0.03],
        [0.01, 0.06, 0.25, 0.58, 0.10],
        [0.20, 0.25, 0.20, 0.15, 0.20],
    ])

    bars: List[Bar] = []
    base_ts = 1_700_000_000

    for i in range(n):
        if reg_timer <= 0:
            regime = int(rng.choice(5, p=T_MAT[regime]))
            lo, hi_d = REG_DUR[regime]
            reg_timer = int(rng.integers(lo, hi_d))
        reg_timer -= 1

        last_ret = ret_hist[-1] if ret_hist else 0.0
        vol = math.sqrt(
            max(OMEGA + ALPHA_G * last_ret ** 2 + BETA_G * vol ** 2, 1e-8))
        eff_vol = float(np.clip(vol * REG_VOL[regime], 0.003, 0.12))

        drift = REG_DRIFT[regime]
        if len(ret_hist) >= 5 and regime in (3, 4):
            drift += float(np.mean(ret_hist[-5:])) * 0.20
        if len(ret_hist) >= 20 and regime == 2:
            drift -= float(np.mean(ret_hist[-20:])) * 0.10

        if rng.random() < 0.015:
            ret = float(rng.normal(drift, eff_vol * 2.5))
        else:
            ret = float(rng.normal(drift, eff_vol))
        ret = float(np.clip(ret, -0.10, 0.10))

        price = float(np.clip(price * math.exp(ret), 1.0, 1_000_000.0))
        ret_hist.append(ret)
        if len(ret_hist) > 100:
            ret_hist.pop(0)

        rng_pct = float(np.clip(abs(rng.normal(0, eff_vol * 1.2)), 0, 0.06))
        spread = price * 0.0003
        gap = float(rng.normal(0, eff_vol * 0.2))
        open_p = float(np.clip(price * math.exp(gap), 1.0, 1_000_000.0))

        if ret >= 0:
            high = price + price * rng_pct * float(rng.uniform(0.5, 1.0))
            low = open_p - price * rng_pct * float(rng.uniform(0.1, 0.4))
        else:
            high = open_p + price * rng_pct * float(rng.uniform(0.1, 0.4))
            low = price - price * rng_pct * float(rng.uniform(0.5, 1.0))

        high = max(high, open_p, price) + spread
        low = min(low,  open_p, price) - spread
        low = max(low, 1e-3)

        fear_mult = 1.3 if ret < -eff_vol else 1.0
        vol_mult = 1.0 + abs(ret) / max(eff_vol, 1e-8) * 0.3
        volume = float(rng.lognormal(
            math.log(max(500_000 * vol_mult * fear_mult, 1.0)), 0.3))

        bars.append(Bar(
            symbol=symbol,
            timeframe=TimeFrame.H1,
            timestamp=base_ts + i * 3600,
            open=float(open_p),
            high=float(high),
            low=float(low),
            close=float(price),
            volume=float(volume),
        ))

    return bars


def augment_bars(bars: List[Bar], noise_std: float = 0.001) -> List[Bar]:
    """Per-iteration noise to prevent price-level memorization."""
    if noise_std <= 0:
        return list(bars)
    rng = np.random.default_rng()
    out: List[Bar] = []
    for b in bars:
        nf = 1.0 + float(rng.normal(0, noise_std))
        out.append(Bar(
            symbol=b.symbol,
            timeframe=b.timeframe,
            timestamp=b.timestamp,
            open=max(float(b.open * nf), 1e-3),
            high=max(float(b.high * nf), 1e-3),
            low=max(float(b.low * nf), 1e-3),
            close=max(float(b.close * nf), 1e-3),
            volume=max(float(b.volume),    1.0),
        ))
    return out


def split_bars(
    bars:      List[Bar],
    train_pct: float = 0.60,
    val_pct:   float = 0.20,
) -> Tuple[List[Bar], List[Bar], List[Bar]]:
    """Strict walk-forward split — never shuffles."""
    n = len(bars)
    t_end = int(n * train_pct)
    v_end = int(n * (train_pct + val_pct))
    return bars[:t_end], bars[t_end:v_end], bars[v_end:]
