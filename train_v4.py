# train_v4.py
"""
Main training script v4.1.
Run:  python train_v4.py

This is the ONLY file you need to run.
All classes from diagnostics.py, reward_engine.py, data_gen.py
are imported here and wired together.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ── repo imports ──────────────────────────────────────────────────────────────
from config import (
    NetworkConfig, PPOConfig, RewardConfig, EnvironmentConfig,
    EnvironmentNotResetError, EpisodeTerminatedError,
    InsufficientDataError, TrainingError,
)
from state import (
    Bar, TimeFrame, Side, ActionSpace, Architecture,
    EpisodeTermination, EpisodeStats, Action,
    Indicators, OrderBookSnapshot, PortfolioSnapshot,
    PositionState, SentimentSnapshot, StepResult, MarketStateBuilder,
)

from utils.torch_utils import select_device

# ── new modules ───────────────────────────────────────────────────────────────
from data_gen import make_bars_v3, augment_bars, split_bars
from reward_engine import RewardEngine, StepContext
from diagnostics import (
    RewardDebugger,
    TrainingMonitor,
    EpisodeTracker,
    CheckpointManager,
)

from networks.actor_critic_network import ActorCriticNetwork

from rollout_buffer import RolloutBuf
from execution_simulator import ExecutionSimulator, ExecutionConfig 


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




# ══════════════════════════════════════════════════════
# Fixed Environment
# ══════════════════════════════════════════════════════

class SingleAssetEnvV4:
    """
    Fixed environment.
    Changes vs trader.py:
    - Uses new RewardEngine from reward_engine.py
    - Stores _original_bars for augmentation
    - Continuous action dead-zone (±0.15 = FLAT)
    - Passes unrealised_pnl_pct to StepContext
    """

    DMAP: Dict[int, Tuple[float, float, float]] = {
        0: (0.0,  0.0,  2.0),
        1: (0.5,  0.3,  2.0),
        2: (1.0,  0.6,  2.0),
        3: (-0.5, 0.3,  2.0),
        4: (-1.0, 0.6,  2.0),
        5: (0.0,  0.0,  0.0),
    }

    def __init__(
        self,
        bars: List[Bar],
        cfg: EnvironmentConfig,
        device: torch.device,
    ) -> None:
        if not bars:
            raise ValueError("bars list is empty")
        self.cfg = cfg
        self.device = device
        self._original_bars = list(bars)   # keep clean copy
        self.bars = list(bars)
        self.inds = [None] * len(bars)
        self.obs_ = [None] * len(bars)
        self.sent = [None] * len(bars)

        self._sb = MarketStateBuilder(
            cfg.lookback_window, device, cfg.normalise_obs, cfg.clip_obs)
        self._re = RewardEngine(cfg.reward, device)
        self._ex = ExecutionSimulator(
            cfg.initial_capital,
            ExecutionConfig(
                cfg.commission_rate, cfg.slippage_rate,
                cfg.funding_rate_per_bar, cfg.max_leverage,
            ),
        )
        self._reset_state()

    def _reset_state(self) -> None:
        self._start_idx = self.cfg.warmup_bars
        self._idx = self._start_idx
        self._dt = 0
        self._pk = self.cfg.initial_capital
        self._done = False
        self._rc = False
        self._term: Optional[EpisodeTermination] = None
        self._stats = EpisodeStats()
        self._max_start = max(
            self.cfg.warmup_bars,
            len(self.bars) - self.cfg.episode_length - self.cfg.warmup_bars,
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
        self._stats = EpisodeStats(
            peak_capital=self.cfg.initial_capital,
            final_capital=self.cfg.initial_capital,
        )
        self._sb.reset()
        self._re.reset()
        self._ex.reset(self.cfg.initial_capital)
        for i in range(
            max(0, self._start_idx - self.cfg.warmup_bars), self._start_idx
        ):
            self._sb.update(self.bars[i])
        return StepResult(*self._obs(), 0.0, False, False, {"reset": True})

    def _parse(self, a: np.ndarray) -> Action:
        if self.cfg.action_space == ActionSpace.DISCRETE:
            idx = max(0, min(int(np.asarray(a).flat[0]), len(self.DMAP) - 1))
            d, s, t = self.DMAP[idx]
            return Action(d, s, t)

        a = np.clip(np.asarray(a, np.float32).flatten(), -1, 1)
        dir_ = float(a[0]) if len(a) > 0 else 0.0

        # Narrower dead-zone: ±0.10 (was ±0.20)
        # At init, action[0] ~ N(0, 0.607) so P(|a|>0.10) ≈ 87%
        # With ±0.20 dead-zone: P(trade) ≈ 71% — but NoisyNet adds variance
        if abs(dir_) < 0.10:
            dir_ = 0.0

        # Size: [0.05, 0.30] — still conservative but not tiny
        raw_size = float((a[1] + 1) / 2) if len(a) > 1 else 0.5
        size = 0.05 + raw_size * 0.25   # [0.05, 0.30]

        # TP multiplier: [1.5, 3.0]
        raw_tp = float((a[2] + 1) / 2) if len(a) > 2 else 0.5
        tp_mult = 1.5 + raw_tp * 1.5

        return Action(dir_, size, tp_mult)
    

    def _obs(self) -> Tuple:
        idx = min(self._idx, len(self.bars) - 1)
        try:
            return self._sb.build(
                self.bars[idx], self.inds[idx],
                self.obs_[idx], self._ex.snap(), self.sent[idx],
            )
        except Exception:
            return self._zero()

    def _zero(self) -> Tuple:
        lw = self.cfg.lookback_window
        return (
            torch.zeros(lw, N_MARKET_FEATURES,    device=self.device),
            torch.zeros(N_PORTFOLIO_FEATURES,      device=self.device),
            torch.zeros(N_SENTIMENT_FEATURES,      device=self.device),
            torch.zeros(N_TIME_FEATURES,           device=self.device),
        )

    def _dd(self) -> float:
        pk = self._pk
        return float(np.clip((pk - self._ex.capital) / pk, 0, 1)) if pk > 0 else 0.0

    def _lev(self) -> float:
        pos = self._ex.position
        cap = self._ex.capital
        return min(pos.notional / cap, 10.0) if pos and cap > 0 else 1.0

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
    def episode_stats(self) -> EpisodeStats: return self._stats


    def _exec(self, bar: Bar, action: Action) -> StepContext:
        """
        FIX: pass position_size_pct to reward engine.
        Also fix the prev_bar index bug.
        """
        price = bar.close
        prev_cap = self._ex.capital
        hit_sl = hit_tp = opened = closed = False
        tp_lvl = 0
        realised_pnl = 0.0
        position_size_pct = 0.0   # track for reward

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
                    price, ts, action.size,
                    action.sl_pct, action.tp_multiplier,
                )
                if not fill.get("skipped"):
                    self._dt += 1
                    self._stats.total_trades += 1
                    opened = True
                    position_size_pct = action.size  # record for reward

        snap = self._ex.snap()
        self._pk = max(self._pk, snap.total_capital)
        total_net = snap.total_capital - prev_cap
        realised_pct = float(np.clip(
            realised_pnl / (prev_cap + 1e-10), -1, 1)) if closed else 0.0
        total_pct = float(np.clip(total_net / (prev_cap + 1e-10), -1, 1))
        unrealised_pct = float(np.clip(
            snap.unrealised_pnl / (snap.total_capital + 1e-10), -1, 1))

        pos = self._ex.position
        ps = 1 if (pos and pos.side == Side.LONG) else -1 if pos else 0

        # FIX: correct prev bar index
        prev_idx = max(0, self._idx - 1)
        pb = self.bars[prev_idx].close
        br = float(np.clip((bar.close - pb) / (pb + 1e-10), -0.5, 0.5))

        ctx = StepContext(
            realised_pnl=realised_pnl,
            realised_pnl_pct=realised_pct,
            total_pnl_pct=total_pct,
            unrealised_pnl_pct=unrealised_pct,
            hit_sl=hit_sl, hit_tp=hit_tp, tp_level=tp_lvl,
            closed=closed, opened=opened,
            drawdown=self._dd(), leverage=self._lev(),
            hold_bars=pos.hold_bars if pos else 0,
            daily_trades=self._dt,
            portfolio_value=snap.total_capital,
            position_side=ps, price_return=br,
        )
        # Store size for reward
        self._last_size = position_size_pct
        return ctx

    def step(self, action: np.ndarray) -> StepResult:
        """Override step to pass position_size_pct to reward engine."""
        if not self._rc:
            raise EnvironmentNotResetError()
        if self._done:
            raise EpisodeTerminatedError(
                self._term.value if self._term else "?")
        if self._idx >= len(self.bars):
            self._done = True
            self._term = EpisodeTermination.DATA_EXHAUSTED
            return StepResult(*self._zero(), 0.0, False, True, {}, self._term)

        bar = self.bars[self._idx]
        self._sb.update(bar)
        self._last_size = 0.0
        ctx = self._exec(bar, self._parse(action))

        # Pass position size to reward engine
        rew, comps = self._re.step(ctx, position_size_pct=self._last_size)
        rew = float(np.clip(rew, -10.0, 10.0))   # tighter clip

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
            snap = self._ex.snap()
            self._stats.final_capital = snap.total_capital
            self._stats.max_drawdown = self._dd()
            self._stats.termination = self._term

        info = {
            "capital":           self._ex.capital,
            "drawdown":          self._dd(),
            "step":              self._idx,
            "daily_trades":      self._dt,
            "reward_components": comps,
            "position":          self._ex.position is not None,
            "termination":       self._term.value if self._term else None,
        }
        if self._done:
            info["episode_stats"] = self._stats

        return StepResult(*self._obs(), rew, term, trunc, info, self._term)

# ══════════════════════════════════════════════════════
# PPO Trainer V4
# ══════════════════════════════════════════════════════

class PPOTrainerV4:
    """
    PPO trainer with all fixes applied.
    Works with diagnostics classes.
    """

    def __init__(
        self,
        net:    ActorCriticNetwork,
        cfg:    PPOConfig,
        dev:    torch.device,
        reward_debugger:  Optional[RewardDebugger] = None,
        training_monitor: Optional[TrainingMonitor] = None,
        episode_tracker:  Optional[EpisodeTracker] = None,
    ) -> None:
        self.net = net.to(dev)
        self.cfg = cfg
        self.device = dev
        self._step = 0
        self._ep_rewards: List[float] = []

        # ── diagnostic tools ──────────────────────────────────────
        self.reward_debugger = reward_debugger
        self.training_monitor = training_monitor
        self.episode_tracker = episode_tracker

        # ── optimizer with encoder / non-encoder split LR ─────────
        pg = [
            {"params": net.non_enc_params(), "lr": cfg.learning_rate},
            {"params": net.enc_params(),     "lr": cfg.learning_rate * 0.3},
        ]
        self.opt = optim.Adam(pg, eps=1e-5)

        def lr_fn(step: int) -> float:
            if step < cfg.warmup_steps:
                return (step + 1) / cfg.warmup_steps
            return max(0.01, 1.0 - (step - cfg.warmup_steps) / 2_000_000)

        self.sched = optim.lr_scheduler.LambdaLR(self.opt, lr_fn)
        self.buf = RolloutBuf(dev)
        self._T: Optional[Dict] = None

    # ── rollout ───────────────────────────────────────────────────

    @torch.no_grad()
    def collect_rollout(self, env: "SingleAssetEnvV4") -> Dict[str, float]:
        self.net.eval()
        self.buf.clear()
        res = env.reset()
        ep_r = 0.0
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
            self.buf.add(res, act, lp, val,
                         reward=nxt.reward, done=float(nxt.done))

            if (self.reward_debugger is not None
                    and "reward_components" in nxt.info):
                self.reward_debugger.log(
                    nxt.info["reward_components"], nxt.reward)

            if (self.episode_tracker is not None
                    and nxt.done and "episode_stats" in nxt.info):
                self.episode_tracker.log_from_info(nxt.info)

            ep_r += nxt.reward
            res = nxt

            if res.done:
                self._ep_rewards.append(ep_r)
                if self.reward_debugger is not None:
                    # FIX: correct attribute name
                    self.reward_debugger.ep_rewards.append(ep_r)
                ep_r = 0.0
                ep_n += 1
                res = env.reset()
                hx = None

        mkt = res.obs_market_seq.unsqueeze(0).to(self.device)
        port = res.obs_portfolio_vec.unsqueeze(0).to(self.device)
        sent = res.obs_sentiment_vec.unsqueeze(0).to(self.device)
        tv = res.obs_time_vec.unsqueeze(0).to(self.device)
        lv = self.net.get_value(mkt, port, sent, tv).squeeze()
        T = self.buf.tensors()
        self.buf.gae(
            T, lv, res.done,
            self.cfg.gamma, self.cfg.gae_lambda,
            self.cfg.normalize_advantage, self.cfg.normalize_returns,
        )
        self._T = T
        self._step += self.cfg.n_steps

        recent = self._ep_rewards[-10:] if self._ep_rewards else [0.0]
        return {"mean_reward": float(np.mean(recent)), "n_episodes": ep_n}


    # ── update ────────────────────────────────────────────────────
    def update(self) -> Dict[str, float]:
        if self._T is None:
            raise TrainingError("call collect_rollout first")
        self.net.train()

        f = min(self._step / max(self.cfg.entropy_anneal_steps, 1), 1.0)
        ec = (self.cfg.entropy_coef
              + (self.cfg.entropy_coef_min - self.cfg.entropy_coef) * f)

        stats: Dict[str, List[float]] = {
            k: [] for k in ("pl", "vl", "ent", "tl", "kl", "cf", "gn")
        }
        early = False

        for _ in range(self.cfg.n_epochs):
            if early:
                break
            kls: List[float] = []
            for b in self.buf.batches(self._T, self.cfg.batch_size):
                self.net.reset_noise()
                ao, co, _ = self.net.get_action_and_value(
                    b["mkt"], b["port"], b["sent"], b["tv"], b["act"])
                lr = (ao.log_prob - b["olp"]).clamp(-20, 20)
                ratio = lr.exp()
                adv = b["adv"]
                ret = b["ret"]
                pl = -torch.min(
                    ratio * adv,
                    ratio.clamp(1 - self.cfg.clip_epsilon,
                                1 + self.cfg.clip_epsilon) * adv,
                ).mean()
                vc = b["oval"] + (co.value - b["oval"]).clamp(
                    -self.cfg.clip_vf_epsilon, self.cfg.clip_vf_epsilon)
                vl = 0.5 * torch.max(
                    F.huber_loss(co.value, ret, reduction="none"),
                    F.huber_loss(vc, ret, reduction="none"),
                ).mean()
                loss = pl + self.cfg.value_coef * vl - ec * ao.entropy.mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.cfg.max_grad_norm)
                self.opt.step()
                self.sched.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - lr).mean().item()
                    cf = ((ratio - 1).abs() >
                          self.cfg.clip_epsilon).float().mean().item()
                for k, v in [
                    ("pl", pl.item()), ("vl", vl.item()),
                    ("ent", ao.entropy.mean().item()), ("tl", loss.item()),
                    ("kl", kl), ("cf", cf), ("gn", float(gn)),
                ]:
                    stats[k].append(v)
                kls.append(kl)
            if (self.cfg.target_kl and kls
                    and float(np.mean(kls)) > self.cfg.target_kl):
                early = True

        result = {
            "policy_loss": float(np.mean(stats["pl"])),
            "value_loss":  float(np.mean(stats["vl"])),
            "entropy":     float(np.mean(stats["ent"])),
            "total_loss":  float(np.mean(stats["tl"])),
            "approx_kl":   float(np.mean(stats["kl"])),
            "clip_frac":   float(np.mean(stats["cf"])),
            "grad_norm":   float(np.mean(stats["gn"])),
            "lr":          self.opt.param_groups[0]["lr"],
        }

        # ── log to monitor ────────────────────────────────────────
        if self.training_monitor is not None:
            self.training_monitor.log(result)

        return result

    # ── evaluation ────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(
        self,
        env: SingleAssetEnvV4,
        n:   int = 5,
        det: bool = True,
    ) -> Dict[str, float]:
        self.net.eval()
        rews, caps, trades, wrs = [], [], [], []
        for ep in range(n):
            r = env.reset(seed=ep)
            er = 0.0
            while not env.is_done:
                r = env.step(self._select_action(r, det))
                er += r.reward
            s = env.episode_stats
            rews.append(er)
            caps.append(s.final_capital)
            trades.append(s.total_trades)
            wrs.append(s.win_rate)
        return {
            "mean_reward":   float(np.mean(rews)),
            "std_reward":    float(np.std(rews)),
            "mean_capital":  float(np.mean(caps)),
            "mean_trades":   float(np.mean(trades)),
            "mean_win_rate": float(np.mean(wrs)),
        }

    @torch.no_grad()
    def _select_action(self, r: StepResult, det: bool) -> np.ndarray:
        self.net.eval()
        mkt = r.obs_market_seq.unsqueeze(0).to(self.device)
        port = r.obs_portfolio_vec.unsqueeze(0).to(self.device)
        sent = r.obs_sentiment_vec.unsqueeze(0).to(self.device)
        tv = r.obs_time_vec.unsqueeze(0).to(self.device)
        ao, _, _ = self.net.get_action_and_value(
            mkt, port, sent, tv, det=det)
        return ao.action.squeeze(0).cpu().numpy()


def main() -> None:
    set_seed(42)
    device = select_device()
    W = 65

    print("=" * W)
    print(f" ZeroStrike DRL v4.2-fixed | device={device}")
    print("=" * W)

    # [1] Data
    print("\n[1/5] Generating data ...")
    all_bars = make_bars_v3(n=10000, seed=42)
    train_bars, val_bars, test_bars = split_bars(all_bars, 0.60, 0.20)
    print(f"      train={len(train_bars)}"
          f"  val={len(val_bars)}"
          f"  test={len(test_bars)}")

    # [2] Configs — smaller network, conservative settings
    print("\n[2/5] Building network ...")
    ncfg = NetworkConfig(
        architecture=Architecture.ATTENTION_LSTM,
        d_model=64, n_heads=4, n_layers=2, d_ff=128,
        lstm_hidden=64, lstm_layers=1, lstm_dropout=0.0,
        hidden_dims=(128, 64), dropout=0.10,
        use_noisy_net=True, use_dueling=True,
    )
    reward_cfg = RewardConfig()
    train_ecfg = EnvironmentConfig(
        episode_length=200, warmup_bars=60,
        initial_capital=100_000.0, random_start=True,
        reward=reward_cfg, max_drawdown_terminate=0.40,
    )
    # BUG 3 FIX: val env also uses random_start=True
    # so each evaluate() call sees different windows
    val_ecfg = EnvironmentConfig(
        episode_length=200, warmup_bars=60,
        initial_capital=100_000.0, random_start=True,
        reward=reward_cfg, max_drawdown_terminate=0.40,
    )
    pcfg = PPOConfig()   # uses defaults from fixed config.py

    net = ActorCriticNetwork(
        ncfg, N_MARKET_FEATURES, N_PORTFOLIO_FEATURES,
        N_SENTIMENT_FEATURES, N_TIME_FEATURES,
        ad=3, cont=True,
    ).to(device)
    pc = net.param_count()
    print(f"      params={pc['total']:,}")

    # Quick forward test
    with torch.no_grad():
        _ao, _co, _ = net.get_action_and_value(
            torch.randn(2, 60, N_MARKET_FEATURES,    device=device),
            torch.randn(2, N_PORTFOLIO_FEATURES,     device=device),
            torch.randn(2, N_SENTIMENT_FEATURES,     device=device),
            torch.randn(2, N_TIME_FEATURES,          device=device),
        )
    print(f"      forward OK | std={float(_ao.std.mean()):.3f}"
          f"  val=[{float(_co.value.min()):.2f},{float(_co.value.max()):.2f}]")

    # [3] Diagnostics
    print("\n[3/5] Diagnostics ...")
    reward_debugger = RewardDebugger(window=500)
    training_monitor = TrainingMonitor(window=20)
    episode_tracker = EpisodeTracker(window=30)
    ckpt_manager = CheckpointManager("/tmp/drl_v42.pt")

    # [4] Training
    print("\n[4/5] Training ...")
    val_env = SingleAssetEnvV4(val_bars, val_ecfg, device)
    trainer = PPOTrainerV4(
        net, pcfg, device,
        reward_debugger=reward_debugger,
        training_monitor=training_monitor,
        episode_tracker=episode_tracker,
    )

    N_ITER = 300
    EVAL_FREQ = 10
    PATIENCE = 30
    NOISE_STD = 0.001

    best_eval = -float("inf")
    patience_cnt = 0

    print(f"\n  {'it':>4}  {'train_R':>8}  {'eval_R':>8}"
          f"  {'cap':>9}  {'wr':>6}  {'trd':>5}"
          f"  {'KL':>7}  {'ent':>6}  status")
    print(f"  {'─' * 72}")

    for it in range(N_ITER):
        aug = augment_bars(train_bars, NOISE_STD)
        t_env = SingleAssetEnvV4(aug, train_ecfg, device)
        roll = trainer.collect_rollout(t_env)
        upd = trainer.update()

        if it % EVAL_FREQ == 0:
            em = trainer.evaluate(val_env, n=5, det=True)
            eval_r = em["mean_reward"]

            saved = ckpt_manager.save(
                net, trainer.opt, trainer.sched,
                trainer._step, eval_r,
            )
            if saved:
                patience_cnt = 0
                status = "↑ best"
            else:
                patience_cnt += 1
                status = f"p={patience_cnt}/{PATIENCE}"

            print(
                f"  {it:>4}  {roll['mean_reward']:>+8.3f}"
                f"  {eval_r:>+8.3f}"
                f"  {em['mean_capital']:>9,.0f}"
                f"  {em['mean_win_rate']:>5.1%}"
                f"  {em['mean_trades']:>5.0f}"
                f"  {upd['approx_kl']:>7.4f}"
                f"  {upd['entropy']:>6.3f}"
                f"  {status}"
            )

            # Show diagnostics after warmup
            if it >= 30:
                t_issues = training_monitor.check()
                e_issues = episode_tracker.check()
                all_iss = t_issues + e_issues
                if all_iss:
                    for iss in all_iss[:2]:
                        print(f"    {iss.strip()}")

            if patience_cnt >= PATIENCE:
                print(f"\n  ⏹ Early stop at iter {it}")
                break

    ckpt_manager.restore(net, trainer.opt, trainer.sched, device)

    # [5] Final
    print("\n[5/5] Final analysis ...")
    print(reward_debugger.report())
    print(training_monitor.summary())
    print(episode_tracker.summary())

    test_ecfg = EnvironmentConfig(
        episode_length=300, warmup_bars=60,
        initial_capital=100_000.0, random_start=False,
        reward=reward_cfg,
    )
    test_env = SingleAssetEnvV4(test_bars, test_ecfg, device)
    em = trainer.evaluate(test_env, n=5, det=True)

    print(f"\n  ── Test Results ──────────────────────────────")
    print(f"  reward:   {em['mean_reward']:+.4f} ± {em['std_reward']:.4f}")
    print(f"  capital:  {em['mean_capital']:,.0f}"
          f"  ({'profit' if em['mean_capital'] > 100_000 else 'loss'})")
    print(f"  win_rate: {em['mean_win_rate']:.1%}")
    print(f"  trades:   {em['mean_trades']:.0f}")

    print(f"\n{'=' * W}")
    if em["mean_capital"] > 102_000:
        print(" ✓ Agent is profitable")
    elif em["mean_win_rate"] > 0.40:
        print(" ~ Learning but not yet profitable — run more iterations")
    else:
        print(" ✗ Not converged — check diagnostics above")
    print("=" * W)


if __name__ == "__main__":
    main()


