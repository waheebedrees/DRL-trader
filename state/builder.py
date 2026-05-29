# zerostrike/drl/state/builder.py

from __future__ import annotations

import datetime
import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn 

from torch import Tensor

from state.features import N_MARKET_FEATURES, N_SENTIMENT_FEATURES, N_PORTFOLIO_FEATURES, N_TIME_FEATURES
from state.normaliser import WelfordNorm
from state.models import Bar, Indicators, OrderBookLevel, OrderBookSnapshot, PortfolioSnapshot, SentimentSnapshot
from config.exceptions import InsufficientDataError
from state.enums import Side

class TorchObservation:
    """
    All observation tensors on the same device.
    Avoids CPU→GPU copies at training time.
    """

    __slots__ = (
        "market_seq", "portfolio_vec",
        "sentiment_vec", "time_vec", "device",
    )

    def __init__(
        self,
        market_seq:    Tensor,   # [T, F]
        portfolio_vec: Tensor,   # [P]
        sentiment_vec: Tensor,   # [S]
        time_vec:      Tensor,   # [6]
        device:        torch.device,
    ) -> None:
        self.market_seq = market_seq
        self.portfolio_vec = portfolio_vec
        self.sentiment_vec = sentiment_vec
        self.time_vec = time_vec
        self.device = device

    def to(self, device: torch.device) -> "TorchObservation":
        return TorchObservation(
            market_seq=self.market_seq.to(device),
            portfolio_vec=self.portfolio_vec.to(device),
            sentiment_vec=self.sentiment_vec.to(device),
            time_vec=self.time_vec.to(device),
            device=device,
        )

    def unsqueeze_batch(self) -> "TorchObservation":
        """Add batch dimension for single inference."""
        return TorchObservation(
            market_seq=self.market_seq.unsqueeze(0),
            portfolio_vec=self.portfolio_vec.unsqueeze(0),
            sentiment_vec=self.sentiment_vec.unsqueeze(0),
            time_vec=self.time_vec.unsqueeze(0),
            device=self.device,
        )

    @property
    def flat(self) -> Tensor:
        return torch.cat([
            self.market_seq.flatten(),
            self.portfolio_vec,
            self.sentiment_vec,
            self.time_vec,
        ])

    def numpy_flat(self):
        return self.flat.cpu().numpy()


class MarketStateBuilderV1:
    """
    Converts raw market data into normalised torch tensors.

    All intermediate computations use torch — GPU-compatible,
    gradient-free (torch.no_grad throughout).
    Normalisation uses Welford's algorithm with torch tensors.
    """

    def __init__(
        self,
        lookback:  int = 60,
        device:    torch.device = torch.device("cpu"),
        normalise: bool = True,
        clip:      float = 5.0,
    ) -> None:
        self.lookback = lookback
        self.device = device
        self.normalise = normalise
        self.clip = clip

        # Bar history: symbol → list of feature tensors [N_MARKET_FEATURES]
        self._history: Dict[str, Deque[Tensor]] = {}

        # Online normalisers (Welford, torch-based)
        self._market_norm = WelfordNorm(N_MARKET_FEATURES, device)
        self._portfolio_norm = WelfordNorm(N_PORTFOLIO_FEATURES, device)

        # Realised vol window
        self._vol_window = 20

    # ── Public API ────────────────────────────────────────────────────

    def update(self, bar: Bar) -> None:
        """Compute feature tensor for bar and append to history."""
        if bar.symbol not in self._history:
            self._history[bar.symbol] = deque(maxlen=self.lookback + 1)

        # Get previous bar for return computation
        hist = self._history[bar.symbol]
        # [0] = close price stashed
        prev_close = float(hist[-1][0]) if hist else bar.close

        feat = self._bar_to_features(bar, prev_close)
        self._history[bar.symbol].append(feat)

    def build(
        self,
        bar:       Bar,
        ind:       Optional[Indicators] = None,
        ob:        Optional[OrderBookSnapshot] = None,
        portfolio: Optional[PortfolioSnapshot] = None,
        sentiment: Optional[SentimentSnapshot] = None,
    ) -> TorchObservation:
        hist = list(self._history.get(bar.symbol, []))
        if len(hist) < 2:
            raise InsufficientDataError(required=2, available=len(hist))

        with torch.no_grad():
            market_seq = self._build_market_seq(hist, ind, ob)
            portfolio_vec = self._build_portfolio_vec(portfolio)
            sentiment_vec = self._build_sentiment_vec(sentiment)
            time_vec = self._build_time_vec(bar.timestamp)

        return TorchObservation(
            market_seq=market_seq,
            portfolio_vec=portfolio_vec,
            sentiment_vec=sentiment_vec,
            time_vec=time_vec,
            device=self.device,
        )

    def reset(self) -> None:
        self._history.clear()

    # ── Market sequence ───────────────────────────────────────────────

    def _bar_to_features(self, bar: Bar, prev_close: float) -> Tensor:
        """
        Convert a single bar to raw feature tensor.
        All maths in torch — keeps everything on device.
        """
        close = bar.close
        # Stash close as first element for next bar's log-return
        log_ret = math.log(close / (prev_close + 1e-8))
        hl_range = (bar.high - bar.low) / (close + 1e-8)
        close_pos = (close - bar.low) / (bar.range +
                                         1e-8) if bar.range > 0 else 0.5
        gap = (bar.open - prev_close) / (prev_close + 1e-8)
        vol_raw = math.log1p(bar.volume)

        # Raw feature vector — indicators filled as 0 here, overwritten in build()
        raw = [
            close,         # [0] stashed for next bar
            log_ret,       # [1] price features start
            hl_range,
            close_pos,
            gap,
            vol_raw,       # [5] volume raw
            0.0, 0.0,      # vol_ratio, vol_mom (need history)
            # Momentum (needs indicators)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # Trend
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # Volatility
            0.0, 0.0, 0.0, 0.0,
            # Volume indicators
            0.0, 0.0, 0.0, 0.0,
            # Microstructure
            0.0, 0.0, 0.0,
            # Regime
            0.0, 0.0, 0.0,
        ]
        return torch.tensor(raw, dtype=torch.float32, device=self.device)

    def _build_market_seq(
        self,
        hist: List[Tensor],
        ind:  Optional[Indicators],
        ob:   Optional[OrderBookSnapshot],
    ) -> Tensor:
        """
        Build [lookback, N_MARKET_FEATURES] tensor.
        Fills volume ratios from history, indicator features for last bar.
        """
        n_avail = len(hist)
        seq = torch.zeros(
            self.lookback, N_MARKET_FEATURES,
            dtype=torch.float32, device=self.device,
        )

        bars_to_use = hist[-self.lookback:]
        offset = self.lookback - len(bars_to_use)

        # ── Volume rolling stats ───────────────────────────────────────
        vol_raws = torch.stack([h[5] for h in bars_to_use])  # log volume
        vol_mean = vol_raws.mean()
        vol_std = vol_raws.std() + 1e-8

        # ── Realised volatility ────────────────────────────────────────
        log_rets = torch.stack([h[1] for h in bars_to_use])  # log returns
        if len(log_rets) >= 2:
            rv_windows = log_rets.unfold(
                0, min(self.lookback, len(log_rets)), 1)
            realised_vol = rv_windows.std(dim=-1)
        else:
            realised_vol = torch.zeros(1, device=self.device)

        for i, feat in enumerate(bars_to_use):
            row_idx = offset + i
            row = seq[row_idx]

            # Price features [1:5]
            row[0] = feat[1]    # log_ret
            row[1] = feat[2]    # hl_range
            row[2] = feat[3]    # close_pos
            row[3] = feat[4]    # gap

            # Volume features [4:7]
            vol_raw = feat[5]
            vol_ratio = (vol_raw - vol_mean) / vol_std
            vol_mom = vol_raw - (
                vol_raws[max(0, i-5):i+1].mean()
                if i > 0 else vol_raw
            )
            row[4] = torch.tanh(vol_ratio)
            row[5] = torch.tanh(vol_mom)
            row[6] = torch.tanh(vol_raw - vol_mean)

            # Realised vol (last available)
            if i < len(realised_vol):
                row[22] = torch.tanh(realised_vol[i] * 100)

        # ── Indicator features (last bar only) ────────────────────────
        last_idx = offset + len(bars_to_use) - 1
        if ind is not None and last_idx >= 0:
            row = seq[last_idx]
            self._fill_indicator_features(row, ind)

        # ── Order book features (last bar only) ───────────────────────
        if ob is not None and last_idx >= 0:
            row = seq[last_idx]
            row[27] = torch.tensor(
                min((ob.spread_bps or 0.0), 200.0) / 200.0,
                device=self.device,
            )
            row[28] = torch.tensor(ob.imbalance, device=self.device)
            depth = ob.bid_depth() / (ob.ask_depth() + 1e-8)
            row[29] = torch.tanh(
                torch.tensor(depth, device=self.device).log()
            )

        # ── Normalise ──────────────────────────────────────────────────
        if self.normalise:
            last_row = seq[last_idx:last_idx+1]           # [1, F]
            self._market_norm.update(last_row)
            mean, std = self._market_norm.stats()
            if mean is not None:
                seq = (seq - mean) / (std + 1e-8)
                seq = seq.clamp(-self.clip, self.clip)

        return seq

    def _fill_indicator_features(self, row: Tensor, ind: Indicators) -> None:
        """In-place fill of indicator features into a single row tensor."""
        def _tanh_scale(v, scale=100.0):
            return math.tanh(v / scale) if v is not None else 0.0

        def _norm_50(v):
            return (v - 50.0) / 50.0 if v is not None else 0.0

        def _norm_rsi(v):
            return (v - 50.0) / 50.0 if v is not None else 0.0

        def _price_dist(level):
            if level is None:
                return 0.0
            ref = float(row[0]) if row[0] != 0 else 1.0
            return math.tanh((ref - level) / (ref + 1e-8) * 10.0)

        # Momentum [7:15]
        row[7] = _norm_rsi(ind.rsi_14)
        row[8] = _tanh_scale(ind.macd_line, 200.0)
        row[9] = _tanh_scale(ind.macd_hist, 100.0)
        row[10] = _norm_50(ind.stoch_k)
        row[11] = _norm_50(ind.stoch_d)
        row[12] = (ind.williams_r or 0.0) / 100.0
        row[13] = _tanh_scale(ind.cci, 100.0)
        row[14] = _norm_rsi(ind.mfi)

        # Trend [15:22]
        row[15] = (ind.adx or 0.0) / 100.0
        row[16] = (ind.adx_plus_di or 0.0) / 100.0
        row[17] = (ind.adx_minus_di or 0.0) / 100.0
        row[18] = _price_dist(ind.ema_12)
        row[19] = _price_dist(ind.ema_26)
        row[20] = _price_dist(ind.sma_50)
        row[21] = _price_dist(ind.sma_200)

        # Volatility [22:26]
        row[23] = _tanh_scale(ind.atr_14, 500.0)
        bb_pct = ind.bb_pct or 0.5
        row[24] = (bb_pct - 0.5) * 2.0
        bb_width = ind.bb_width or 0.0
        row[25] = _tanh_scale(bb_width, 2000.0)

        # Volume indicators [26:30] (microstructure starts 27)
        row[26] = _tanh_scale(ind.cmf, 1.0)

        # Regime [30:33]
        row[30] = (ind.adx or 0.0) / 100.0           # trend_strength
        row[31] = _tanh_scale(ind.atr_14, 500.0)      # vol_regime
        row[32] = (                                    # momentum composite
            _norm_rsi(ind.rsi_14) * 0.4
            + _tanh_scale(ind.macd_line, 200.0) * 0.6
        )

    # ── Portfolio vector ──────────────────────────────────────────────

    def _build_portfolio_vec(
        self, portfolio: Optional[PortfolioSnapshot]
    ) -> Tensor:
        vec = torch.zeros(N_PORTFOLIO_FEATURES,
                          dtype=torch.float32, device=self.device)

        if portfolio is None:
            return vec

        cap = portfolio.total_capital + 1e-8
        pos = portfolio.position

        if pos is not None:
            vec[0] = 1.0 if pos.side == Side.LONG else -1.0
            vec[1] = pos.notional / cap
            vec[2] = pos.unrealised_pnl / cap
            vec[3] = pos.max_drawdown
            vec[7] = math.tanh(pos.hold_bars / 100.0)
            vec[8] = math.tanh(pos.dist_to_sl * 10.0)
            vec[9] = math.tanh(pos.dist_to_tp * 10.0)

        vec[4] = portfolio.available_cash / cap
        vec[5] = portfolio.exposure_pct
        vec[6] = portfolio.portfolio_heat

        if self.normalise:
            self._portfolio_norm.update(vec.unsqueeze(0))
            mean, std = self._portfolio_norm.stats()
            if mean is not None:
                vec = ((vec - mean.squeeze(0)) / (std.squeeze(0) + 1e-8)).clamp(
                    -self.clip, self.clip
                )

        return vec

    # ── Sentiment ─────────────────────────────────────────────────────

    def _build_sentiment_vec(
        self, sentiment: Optional[SentimentSnapshot]
    ) -> Tensor:
        if sentiment is None:
            return torch.zeros(N_SENTIMENT_FEATURES, dtype=torch.float32, device=self.device)
        return torch.tensor([
            max(-1.0, min(1.0, sentiment.overall)),
            max(0.0, min(1.0, sentiment.confidence)),
            max(-1.0, min(1.0, sentiment.news)),
            max(-1.0, min(1.0, sentiment.social)),
            max(-1.0, min(1.0, sentiment.options_flow)),
        ], dtype=torch.float32, device=self.device)

    # ── Time encoding ─────────────────────────────────────────────────

    def _build_time_vec(self, timestamp: float) -> Tensor:
        try:
            dt = datetime.datetime.fromtimestamp(
                timestamp, tz=datetime.timezone.utc)
        except Exception:
            dt = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
        tau = 2.0 * math.pi
        return torch.tensor([
            math.sin(tau * dt.hour / 24),
            math.cos(tau * dt.hour / 24),
            math.sin(tau * dt.weekday() / 7),
            math.cos(tau * dt.weekday() / 7),
            math.sin(tau * dt.month / 12),
            math.cos(tau * dt.month / 12),
        ], dtype=torch.float32, device=self.device)



class MarketStateBuilder(nn.Module):
    def __init__(self, lookback: int, device: torch.device,
                 normalise: bool = True, clip: float = 5.0) -> None:
        super().__init__()
        self.lookback = lookback
        self.device = device
        self.normalise = normalise
        self.clip = clip
        self._close: Dict[str, Deque[float]] = {}
        self._feats: Dict[str, Deque[Tensor]] = {}
        self.mkt_norm = WelfordNorm(N_MARKET_FEATURES,    device)
        self.port_norm = WelfordNorm(N_PORTFOLIO_FEATURES, device)

    def update(self, bar: Bar) -> None:
        sym = bar.symbol
        if sym not in self._close:
            self._close[sym] = deque(maxlen=self.lookback + 2)
            self._feats[sym] = deque(maxlen=self.lookback + 2)
        ch = self._close[sym]
        prev = ch[-1] if ch else bar.close
        ch.append(bar.close)
        self._feats[sym].append(self._row(bar, prev))

    def reset(self) -> None:
        self._close.clear()
        self._feats.clear()

    @torch.no_grad()
    def build(self, bar: Bar,
              ind:  Optional[Indicators] = None,
              ob:   Optional[OrderBookSnapshot] = None,
              port: Optional[PortfolioSnapshot] = None,
              sent: Optional[SentimentSnapshot] = None,
              ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        hist = list(self._feats.get(bar.symbol, []))
        if len(hist) < 2:
            raise InsufficientDataError(2, len(hist))
        return (
            self._mkt_seq(hist, bar.symbol, ind, ob),
            self._port_vec(port),
            self._sent_vec(sent),
            self._time_vec(bar.timestamp),
        )

    def _row(self, bar: Bar, prev: float) -> Tensor:
        c = max(bar.close, 1e-10)
        p = max(prev, 1e-10)
        row = torch.zeros(N_MARKET_FEATURES,
                          dtype=torch.float32, device=self.device)
        row[0] = float(math.log(c / p))
        row[1] = float(bar.bar_range / (c + 1e-10))
        row[2] = float(bar.close_position)
        row[3] = float((bar.open - p) / (p + 1e-10))
        row[6] = float(math.log1p(max(bar.volume, 0)))
        return row

    @torch.no_grad()
    def _mkt_seq(self, hist: List[Tensor], sym: str,
                 ind: Optional[Indicators],
                 ob:  Optional[OrderBookSnapshot]) -> Tensor:
        seq = torch.zeros(self.lookback, N_MARKET_FEATURES,
                          dtype=torch.float32, device=self.device)
        bars = hist[-self.lookback:]
        n = len(bars)
        off = self.lookback - n
        vols = torch.stack([b[6] for b in bars])
        vm = vols.mean()
        vs = vols.std().clamp(1e-6)
        lrs = torch.stack([b[0] for b in bars])
        for i, f in enumerate(bars):
            r = off + i
            seq[r, 0] = f[0]
            seq[r, 1] = f[1]
            seq[r, 2] = f[2]
            seq[r, 3] = f[3]
            seq[r, 4] = (f[6] - vm) / vs
            seq[r, 5] = (f[6] - vols[max(0, i - 5):i + 1].mean()) / vs
            seq[r, 6] = f[6]
            if i >= 1:
                rv = lrs[max(0, i - 19):i + 1].std().clamp(0)
                seq[r, 22] = float(torch.tanh(rv * 50).item())
        last = off + n - 1
        if 0 <= last < self.lookback:
            if ind is not None:
                self._fill_ind(seq[last], ind, sym)
            if ob is not None:
                self._fill_ob(seq[last], ob)
            seq[last, 34] = seq[last, 15].item()
            seq[last, 35] = seq[last, 22].item()
            seq[last, 36] = seq[last, 7].item() * 0.4 + \
                seq[last, 8].item() * 0.6
        valid = seq[off:]
        if valid.size(0) > 0:
            self.mkt_norm.update(valid.detach())
        if self.normalise:
            seq = self.mkt_norm.normalise(seq, self.clip)
        return seq

    def _fill_ind(self, row: Tensor, ind: Indicators, sym: str) -> None:
        def t(v, s=1.): return float(math.tanh(v / s)) if v is not None else 0.
        def n50(v): return (v - 50.) / 50. if v is not None else 0.
        ch = self._close.get(sym, deque())
        ref = max(float(ch[-1]) if ch else 1., 1e-10)
        def pct(lv): return float(math.tanh((lv - ref) / ref * 10)
                                  ) if lv is not None else 0.
        row[7] = n50(ind.rsi_14)
        row[8] = t(ind.macd_line, 200)
        row[9] = t(ind.macd_hist, 100)
        row[10] = n50(ind.stoch_k)
        row[11] = n50(ind.stoch_d)
        row[12] = (ind.williams_r or 0) / 100
        row[13] = t(ind.cci, 100)
        row[14] = n50(ind.mfi)
        row[15] = (ind.adx or 0) / 100
        row[16] = (ind.adx_plus_di or 0) / 100
        row[17] = (ind.adx_minus_di or 0) / 100
        row[18] = pct(ind.ema_12)
        row[19] = pct(ind.ema_26)
        row[20] = pct(ind.sma_50)
        row[21] = pct(ind.sma_200)
        row[23] = t(ind.atr_14, max(ref * .01, 1))
        row[24] = ((ind.bb_pct or .5) - .5) * 2
        row[25] = t(ind.bb_width, max(ref * .02, 1))
        if ind.keltner_upper and ind.keltner_lower:
            row[26] = t(ind.keltner_upper -
                        ind.keltner_lower, max(ref * .02, 1))
        row[27] = t(ind.cmf, 1)
        row[29] = pct(ind.vwap)

    def _fill_ob(self, row: Tensor, ob: OrderBookSnapshot) -> None:
        row[31] = min(ob.spread_bps or 0, 200) / 200
        row[32] = float(ob.imbalance)
        d = ob.bid_depth() / (ob.ask_depth() + 1e-10)
        row[33] = float(math.tanh(math.log(max(d, 1e-10))))

    @torch.no_grad()
    def _port_vec(self, p: Optional[PortfolioSnapshot]) -> Tensor:
        vec = torch.zeros(N_PORTFOLIO_FEATURES,
                          dtype=torch.float32, device=self.device)
        if p is None:
            return vec
        cap = max(p.total_capital, 1e-10)
        pos = p.position
        if pos:
            vec[0] = 1. if pos.side == Side.LONG else -1.
            vec[1] = float(np.clip(pos.notional / cap, 0, 5))
            vec[2] = float(np.clip(pos.unrealised_pnl / cap, -1, 1))
            vec[3] = float(min(pos.max_drawdown, 1))
            vec[7] = float(math.tanh(pos.hold_bars / 50))
            vec[8] = float(math.tanh(pos.dist_to_sl * 20))
            vec[9] = float(math.tanh(pos.dist_to_tp * 20))
        vec[4] = float(np.clip(p.available_cash / cap, 0, 2))
        vec[5] = float(min(p.exposure_pct, 1))
        vec[6] = float(np.clip(p.portfolio_heat, 0, 1))
        if self.normalise:
            self.port_norm.update(vec.unsqueeze(0))
            vec = self.port_norm.normalise(vec, self.clip)
        return vec

    @torch.no_grad()
    def _sent_vec(self, s: Optional[SentimentSnapshot]) -> Tensor:
        if s is None:
            return torch.zeros(N_SENTIMENT_FEATURES, dtype=torch.float32, device=self.device)
        return torch.tensor([
            np.clip(s.overall,      -1, 1), np.clip(s.confidence,   0, 1),
            np.clip(s.news,         -1, 1), np.clip(s.social,       -1, 1),
            np.clip(s.options_flow, -1, 1),
        ], dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _time_vec(self, ts: float) -> Tensor:
        try:
            dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        except Exception:
            dt = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
        tau = 2 * math.pi
        return torch.tensor([
            math.sin(tau * dt.hour / 24), math.cos(tau * dt.hour / 24),
            math.sin(tau * dt.weekday() / 7), math.cos(tau * dt.weekday() / 7),
            math.sin(tau * dt.month / 12), math.cos(tau * dt.month / 12),
        ], dtype=torch.float32, device=self.device)
