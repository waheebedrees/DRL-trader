from __future__ import annotations


import os
import random
import math
from typing import Dict, List, Optional, Tuple, Any
import copy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import (
    NetworkConfig,
    RainbowConfig,
    RewardConfig,
    EnvironmentConfig,
    PPOConfig,
    TD3Config,
    SACConfig,
    TrainingError,

)
from networks import (
    # Layers
    NoisyLinear,
    GatedResidualBlock,
    PositionalEncoding,
    RotaryEmbedding,
    MarketAttention,
    SwiGLUFFN,
    DilatedCausalConv,
    SqueezeExcitation,
    # Encoders
    MarketTransformerEncoder,
    AttentionLSTMEncoder,
    TemporalCNNEncoder,
    HybridEncoder,
    PortfolioEncoder,
    build_encoder,
    # Heads
    ContinuousActor,
    DiscreteActor,
    DuelingCritic,
    DistributionalCritic,
    TwinQCritic,
    # Output containers
    ActorOutput,
    CriticOutput,
    DistributionalOutput,
)

from utils.torch_utils import (
    select_device,

)
from state import (

    EpisodeStats,
    TimeFrame,
    Side,
    EpisodeTermination,
    ActionSpace,
    Architecture,
    Action,
    Indicators,
    OrderBookSnapshot,
    SentimentSnapshot,
    StepResult,
    Bar,
    MarketStateBuilder,
)

from data_gen import make_bars
from back_tester import Backtester, BacktestMetrics
from base_trainer import BaseTrainer
from single_asset_env import SingleAssetEnv, StepContext
from replay import PERBuffer, SumTree, Transition, UniformBuf
from rollout_buffer import RolloutBuf
from networks.actor_critic_network import ActorCriticNetwork

def load_csv_to_bars(
    filepath: str,
    symbol: str = "ASSET",
    timeframe: TimeFrame = TimeFrame.H1,
) -> List[Bar]:
    """Load a CSV file and convert to Bar objects."""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna()

    bars = []
    for idx, row in df.iterrows():
        if isinstance(idx, pd.Timestamp):
            ts = int(idx.timestamp())
        else:
            ts = int(pd.Timestamp(idx).timestamp())

        # Fix minor OHLC inconsistencies
        o = float(row['open'])
        h = max(float(row['high']), o, float(row['close']))
        l = min(float(row['low']), o, float(row['close']))
        c = float(row['close'])
        v = max(float(row['volume']), 1.0)

        bar = Bar(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=ts,
            open=o, high=h, low=l, close=c, volume=v,
        )
        bars.append(bar)

    return bars


def split_bars_chronological(
    bars: List[Bar],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[List[Bar], List[Bar], List[Bar]]:
    """
    Split bars chronologically (no shuffling — preserves time order).
    train | val | test
    """
    n = len(bars)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return bars[:train_end], bars[train_end:val_end], bars[val_end:]


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


class _NoAMP:
    def __enter__(self): return self
    def __exit__(self, *a): pass


class PPOTrainer(BaseTrainer):
    def __init__(self, net: ActorCriticNetwork, cfg: PPOConfig, dev: torch.device) -> None:
        super().__init__(net, dev)
        self.cfg = cfg
        pg = [{"params": net.non_enc_params(), "lr": cfg.learning_rate},
              {"params": net.enc_params(),     "lr": cfg.learning_rate * 0.3}]
        self.opt = optim.Adam(pg, eps=1e-5)

        def lr_fn(step: int) -> float:
            if step < cfg.warmup_steps:
                return (step + 1) / cfg.warmup_steps
            return max(0.01, 1. - (step - cfg.warmup_steps) / 1_000_000)
        self.sched = optim.lr_scheduler.LambdaLR(self.opt, lr_fn)
        self.scaler = (torch.cuda.amp.GradScaler()
                       if cfg.use_amp and dev.type == "cuda" else None)
        self.buf = RolloutBuf(dev)
        self._T: Optional[Dict] = None

    def _ent_c(self) -> float:
        f = min(self._step / max(self.cfg.entropy_anneal_steps, 1), 1.)
        return self.cfg.entropy_coef + (self.cfg.entropy_coef_min - self.cfg.entropy_coef) * f

    @torch.no_grad()
    def collect_rollout(self, env: SingleAssetEnv) -> Dict[str, float]:
        self.net.eval()
        self.buf.clear()
        res = env.reset()
        ep_r = 0.
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
            self.buf.add(res, act, lp, val, reward=nxt.reward,
                         done=float(nxt.done))
            ep_r += nxt.reward
            res = nxt
            if res.done:
                self._ep_rewards.append(ep_r)
                ep_r = 0.
                ep_n += 1
                res = env.reset()
                hx = None
        mkt = res.obs_market_seq.unsqueeze(0).to(self.device)
        port = res.obs_portfolio_vec.unsqueeze(0).to(self.device)
        sent = res.obs_sentiment_vec.unsqueeze(0).to(self.device)
        tv = res.obs_time_vec.unsqueeze(0).to(self.device)
        lv = self.net.get_value(mkt, port, sent, tv).squeeze()
        T = self.buf.tensors()
        self.buf.gae(T, lv, res.done, self.cfg.gamma, self.cfg.gae_lambda,
                     self.cfg.normalize_advantage, self.cfg.normalize_returns)
        self._T = T
        self._step += self.cfg.n_steps
        recent = self._ep_rewards[-10:] if self._ep_rewards else [0.]
        return {"mean_reward": float(np.mean(recent)), "n_episodes": ep_n, "n_steps": len(self.buf)}

    def update(self) -> Dict[str, float]:
        if self._T is None:
            raise TrainingError("call collect_rollout first")
        self.net.train()
        ec = self._ent_c()
        stats: Dict[str, List[float]] = {k: [] for k in (
            "pl", "vl", "ent", "tl", "kl", "cf", "gn")}
        early = False
        for epoch in range(self.cfg.n_epochs):
            if early:
                break
            kls: List[float] = []
            for b in self.buf.batches(self._T, self.cfg.batch_size):
                self.net.reset_noise()
                ctx = torch.cuda.amp.autocast() if self.scaler else _NoAMP()
                with ctx:
                    ao, co, _ = self.net.get_action_and_value(
                        b["mkt"], b["port"], b["sent"], b["tv"], b["act"])
                    lr = (ao.log_prob - b["olp"]).clamp(-20, 20)
                    ratio = lr.exp()
                    adv = b["adv"]
                    ret = b["ret"]
                    surr1 = ratio * adv
                    surr2 = ratio.clamp(1 - self.cfg.clip_epsilon,
                                        1 + self.cfg.clip_epsilon) * adv
                    pl = -torch.min(surr1, surr2).mean()
                    vc = b["oval"] + (co.value - b["oval"]).clamp(
                        -self.cfg.clip_vf_epsilon, self.cfg.clip_vf_epsilon)
                    vl = 0.5 * torch.max(
                        F.huber_loss(co.value, ret, reduction="none"),
                        F.huber_loss(vc,       ret, reduction="none")).mean()
                    loss = pl + self.cfg.value_coef * vl - ec * ao.entropy.mean()
                self.opt.zero_grad(set_to_none=True)
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                    gn = nn.utils.clip_grad_norm_(
                        self.net.parameters(), self.cfg.max_grad_norm)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    loss.backward()
                    gn = nn.utils.clip_grad_norm_(
                        self.net.parameters(), self.cfg.max_grad_norm)
                    self.opt.step()
                self.sched.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - lr).mean().item()
                    cf = ((ratio - 1).abs() >
                          self.cfg.clip_epsilon).float().mean().item()
                for k, v in [("pl", pl.item()), ("vl", vl.item()),
                             ("ent", ao.entropy.mean().item()),
                             ("tl", loss.item()), ("kl", kl), ("cf", cf),
                             ("gn", float(gn))]:
                    stats[k].append(v)
                kls.append(kl)
            if self.cfg.target_kl and kls and float(np.mean(kls)) > self.cfg.target_kl:
                early = True
        return {"policy_loss": float(np.mean(stats["pl"])), "value_loss": float(np.mean(stats["vl"])),
                "entropy":     float(np.mean(stats["ent"])), "total_loss": float(np.mean(stats["tl"])),
                "approx_kl":   float(np.mean(stats["kl"])), "clip_frac":  float(np.mean(stats["cf"])),
                "grad_norm":   float(np.mean(stats["gn"])),
                "lr": self.opt.param_groups[0]["lr"], "entropy_coef": ec}

    @torch.no_grad()
    def _select_action(self, r: StepResult, det: bool) -> np.ndarray:
        self.net.eval()
        mkt = r.obs_market_seq.unsqueeze(0).to(self.device)
        port = r.obs_portfolio_vec.unsqueeze(0).to(self.device)
        sent = r.obs_sentiment_vec.unsqueeze(0).to(self.device)
        tv = r.obs_time_vec.unsqueeze(0).to(self.device)
        ao, _, _ = self.net.get_action_and_value(mkt, port, sent, tv, det=det)
        return ao.action.squeeze(0).cpu().numpy()

    def _extra_save(self) -> Dict:
        return {"opt": self.opt.state_dict(), "sched": self.sched.state_dict()}

    def _extra_load(self, ck: Dict) -> None:
        if "opt" in ck:
            self.opt.load_state_dict(ck["opt"])
        if "sched" in ck:
            self.sched.load_state_dict(ck["sched"])


# ==============================================================================
# SECTION 17: SAC Trainer
# ==============================================================================

class SACTrainer(BaseTrainer):
    def __init__(self, net: ActorCriticNetwork, cfg: SACConfig, dev: torch.device, ad: int) -> None:
        super().__init__(net, dev)
        self.cfg = cfg
        self.ad = ad
        td = net.cfg.hidden_dims[-1]
        self.c1 = TwinQCritic(td, ad).to(dev)
        self.c2 = TwinQCritic(td, ad).to(dev)
        self.tc1 = copy.deepcopy(self.c1)
        self.tc2 = copy.deepcopy(self.c2)
        for p in self.tc1.parameters():
            p.requires_grad_(False)
        for p in self.tc2.parameters():
            p.requires_grad_(False)
        self.aopt = optim.Adam(net.parameters(),     lr=cfg.learning_rate)
        self.c1opt = optim.Adam(self.c1.parameters(), lr=cfg.learning_rate)
        self.c2opt = optim.Adam(self.c2.parameters(), lr=cfg.learning_rate)
        if cfg.auto_alpha:
            self.te = cfg.target_entropy if cfg.target_entropy is not None else - \
                float(ad)
            self.la = torch.zeros(1, requires_grad=True, device=dev)
            self.aaopt = optim.Adam([self.la], lr=cfg.learning_rate)
        else:
            self.la = torch.log(torch.tensor(cfg.alpha, device=dev))
        self.replay = PERBuffer(cfg.buffer_size, dev)

    @property
    def alpha(self) -> torch.Tensor: return self.la.exp()

    @torch.no_grad()
    def _soft(self, s: nn.Module, t: nn.Module) -> None:
        for sp, tp in zip(s.parameters(), t.parameters()):
            tp.data.mul_(1 - self.cfg.tau)
            tp.data.add_(self.cfg.tau * sp.data)

    def update(self) -> Dict[str, float]:
        if not self.replay.is_ready(self.cfg.learning_starts):
            return {"skipped": 1.}
        c1l, c2l, al, all_ = [], [], [], []
        for _ in range(self.cfg.gradient_steps):
            batch, idxs, ws = self.replay.sample(self.cfg.batch_size)
            data, wt = self.replay.collate(batch, ws)
            self.net.eval()
            with torch.no_grad():
                nf, _ = self.net.encode(data["next_market_seqs"], data["next_portfolio_vecs"],
                                        data["next_sentiment_vecs"], data["next_time_vecs"])
                ao_n = self.net.actor(nf)
                qn = torch.min(self.tc1(nf, ao_n.action),
                               self.tc2(nf, ao_n.action)).squeeze(-1)
                qt = data["reward"] + self.cfg.gamma * \
                    (1 - data["done"]) * (qn - self.alpha * ao_n.log_prob)
                cf, _ = self.net.encode(data["market_seqs"], data["portfolio_vecs"],
                                        data["sentiment_vecs"], data["time_vecs"])
            q1 = self.c1(cf, data["actions"]).squeeze(-1)
            q2 = self.c2(cf, data["actions"]).squeeze(-1)
            td_e = ((q1 - qt).abs() + (q2 - qt).abs()).detach() / 2
            l1 = (F.huber_loss(q1, qt, reduction="none") * wt).mean()
            l2 = (F.huber_loss(q2, qt, reduction="none") * wt).mean()
            self.c1opt.zero_grad()
            l1.backward()
            nn.utils.clip_grad_norm_(
                self.c1.parameters(), self.cfg.max_grad_norm)
            self.c1opt.step()
            self.c2opt.zero_grad()
            l2.backward()
            nn.utils.clip_grad_norm_(
                self.c2.parameters(), self.cfg.max_grad_norm)
            self.c2opt.step()
            self.net.train()
            af, _ = self.net.encode(data["market_seqs"], data["portfolio_vecs"],
                                    data["sentiment_vecs"], data["time_vecs"])
            ao_a = self.net.actor(af)
            with torch.no_grad():
                qa = torch.min(self.c1(af.detach(), ao_a.action),
                               self.c2(af.detach(), ao_a.action)).squeeze(-1)
            aloss = (self.alpha.detach() * ao_a.log_prob - qa).mean()
            self.aopt.zero_grad()
            aloss.backward()
            nn.utils.clip_grad_norm_(
                self.net.parameters(), self.cfg.max_grad_norm)
            self.aopt.step()
            al_l = torch.tensor(0., device=self.device)
            if self.cfg.auto_alpha:
                al_l = -(self.la * (ao_a.log_prob.detach() + self.te)).mean()
                self.aaopt.zero_grad()
                al_l.backward()
                self.aaopt.step()
            self.replay.update_priorities(idxs, td_e.cpu().numpy())
            c1l.append(float(l1.item()))
            c2l.append(float(l2.item()))
            al.append(float(aloss.item()))
            all_.append(float(al_l.item()))
        self._soft(self.c1, self.tc1)
        self._soft(self.c2, self.tc2)
        self._step += 1
        return {"critic_loss": float(np.mean(c1l)) + float(np.mean(c2l)),
                "actor_loss":  float(np.mean(al)),  "alpha_loss": float(np.mean(all_)),
                "alpha": float(self.alpha.item())}

    def collect_rollout(self, env: SingleAssetEnv) -> Dict[str, float]:
        self.net.eval()
        res = env.reset()
        er = 0.
        en = 0
        for _ in range(self.cfg.train_freq):
            mkt = res.obs_market_seq.unsqueeze(0).to(self.device)
            port = res.obs_portfolio_vec.unsqueeze(0).to(self.device)
            sent = res.obs_sentiment_vec.unsqueeze(0).to(self.device)
            tv = res.obs_time_vec.unsqueeze(0).to(self.device)
            with torch.no_grad():
                ao, _, _ = self.net.get_action_and_value(mkt, port, sent, tv)
            nr = env.step(ao.action.squeeze(0).cpu().numpy())
            self.replay.add(Transition(
                res.obs_market_seq.cpu().numpy(), res.obs_portfolio_vec.cpu().numpy(),
                res.obs_sentiment_vec.cpu().numpy(), res.obs_time_vec.cpu().numpy(),
                nr.obs_market_seq.cpu().numpy(),  nr.obs_portfolio_vec.cpu().numpy(),
                nr.obs_sentiment_vec.cpu().numpy(), nr.obs_time_vec.cpu().numpy(),
                ao.action.squeeze(0).cpu().numpy(), nr.reward, nr.done, nr.info))
            er += nr.reward
            res = nr
            if res.done:
                self._ep_rewards.append(er)
                er = 0.
                en += 1
                res = env.reset()
        recent = self._ep_rewards[-10:] if self._ep_rewards else [0.]
        return {"mean_reward": float(np.mean(recent)), "n_episodes": en, "buffer_size": len(self.replay)}

    @torch.no_grad()
    def _select_action(self, r: StepResult, det: bool) -> np.ndarray:
        self.net.eval()
        mkt = r.obs_market_seq.unsqueeze(0).to(self.device)
        port = r.obs_portfolio_vec.unsqueeze(0).to(self.device)
        sent = r.obs_sentiment_vec.unsqueeze(0).to(self.device)
        tv = r.obs_time_vec.unsqueeze(0).to(self.device)
        ao, _, _ = self.net.get_action_and_value(mkt, port, sent, tv, det=det)
        return ao.action.squeeze(0).cpu().numpy()


# ==============================================================================
# SECTION 18: TD3 Trainer
# ==============================================================================

class TD3Trainer(BaseTrainer):
    def __init__(self, net: ActorCriticNetwork, cfg: TD3Config, dev: torch.device, ad: int) -> None:
        super().__init__(net, dev)
        self.cfg = cfg
        self.ad = ad
        td = net.cfg.hidden_dims[-1]
        self.c1 = TwinQCritic(td, ad).to(dev)
        self.c2 = TwinQCritic(td, ad).to(dev)
        self.tc1 = copy.deepcopy(self.c1)
        self.tc2 = copy.deepcopy(self.c2)
        self.ta = copy.deepcopy(net)
        for p in self.tc1.parameters():
            p.requires_grad_(False)
        for p in self.tc2.parameters():
            p.requires_grad_(False)
        for p in self.ta.parameters():
            p.requires_grad_(False)
        self.aopt = optim.Adam(net.parameters(), lr=cfg.learning_rate)
        self.copt = optim.Adam(list(self.c1.parameters()) + list(self.c2.parameters()),
                               lr=cfg.learning_rate)
        self.replay = PERBuffer(cfg.buffer_size, dev)

    @torch.no_grad()
    def _soft(self, s: nn.Module, t: nn.Module) -> None:
        for sp, tp in zip(s.parameters(), t.parameters()):
            tp.data.mul_(1 - self.cfg.tau)
            tp.data.add_(self.cfg.tau * sp.data)

    def update(self) -> Dict[str, float]:
        if not self.replay.is_ready(self.cfg.learning_starts):
            return {"skipped": 1.}
        batch, _, _ = self.replay.sample(self.cfg.batch_size)
        data = self.replay.collate(batch, np.ones(self.cfg.batch_size))[0]
        self.net.eval()
        with torch.no_grad():
            nf, _ = self.ta.encode(data["next_market_seqs"], data["next_portfolio_vecs"],
                                   data["next_sentiment_vecs"], data["next_time_vecs"])
            ao_n = self.ta.actor(nf, deterministic=True)
            noise = (torch.randn_like(ao_n.action) * self.cfg.policy_noise).clamp(
                -self.cfg.noise_clip, self.cfg.noise_clip)
            na = (ao_n.action + noise).clamp(-1, 1)
            qt = (data["reward"].unsqueeze(-1) + self.cfg.gamma * (1 - data["done"].unsqueeze(-1))
                  * torch.min(self.tc1(nf, na), self.tc2(nf, na)))
            cf, _ = self.net.encode(data["market_seqs"], data["portfolio_vecs"],
                                    data["sentiment_vecs"], data["time_vecs"])
        cl = (F.huber_loss(self.c1(cf, data["actions"]), qt)
              + F.huber_loss(self.c2(cf, data["actions"]), qt))
        self.copt.zero_grad()
        cl.backward()
        nn.utils.clip_grad_norm_(list(self.c1.parameters()) + list(self.c2.parameters()),
                                 self.cfg.max_grad_norm)
        self.copt.step()
        al = torch.tensor(0., device=self.device)
        if self._step % self.cfg.policy_delay == 0:
            self.net.train()
            af, _ = self.net.encode(data["market_seqs"], data["portfolio_vecs"],
                                    data["sentiment_vecs"], data["time_vecs"])
            ao_a = self.net.actor(af, deterministic=True)
            al = -self.c1(af.detach(), ao_a.action).mean()
            self.aopt.zero_grad()
            al.backward()
            nn.utils.clip_grad_norm_(
                self.net.parameters(), self.cfg.max_grad_norm)
            self.aopt.step()
            self._soft(self.net, self.ta)
            self._soft(self.c1, self.tc1)
            self._soft(self.c2, self.tc2)
        self._step += 1
        return {"critic_loss": float(cl.item()), "actor_loss": float(al.item())}

    def collect_rollout(self, env: SingleAssetEnv) -> Dict[str, float]:
        self.net.eval()
        res = env.reset()
        ec = 0
        er = 0.
        for _ in range(max(self.cfg.learning_starts // 10, 20)):
            mkt = res.obs_market_seq.unsqueeze(0).to(self.device)
            port = res.obs_portfolio_vec.unsqueeze(0).to(self.device)
            sent = res.obs_sentiment_vec.unsqueeze(0).to(self.device)
            tv = res.obs_time_vec.unsqueeze(0).to(self.device)
            with torch.no_grad():
                ao, _, _ = self.net.get_action_and_value(mkt, port, sent, tv)
            noise = torch.randn_like(ao.action) * self.cfg.exploration_noise
            a = (ao.action + noise).clamp(-1, 1).squeeze(0).cpu().numpy()
            nr = env.step(a)
            self.replay.add(Transition(
                res.obs_market_seq.cpu().numpy(), res.obs_portfolio_vec.cpu().numpy(),
                res.obs_sentiment_vec.cpu().numpy(), res.obs_time_vec.cpu().numpy(),
                nr.obs_market_seq.cpu().numpy(),  nr.obs_portfolio_vec.cpu().numpy(),
                nr.obs_sentiment_vec.cpu().numpy(), nr.obs_time_vec.cpu().numpy(),
                a, nr.reward, nr.done, nr.info))
            er += nr.reward
            res = nr
            if res.done:
                self._ep_rewards.append(er)
                ec += 1
                er = 0.
                res = env.reset()
        return {"n_episodes": ec, "buffer_size": len(self.replay)}

    @torch.no_grad()
    def _select_action(self, r: StepResult, det: bool) -> np.ndarray:
        self.net.eval()
        mkt = r.obs_market_seq.unsqueeze(0).to(self.device)
        port = r.obs_portfolio_vec.unsqueeze(0).to(self.device)
        sent = r.obs_sentiment_vec.unsqueeze(0).to(self.device)
        tv = r.obs_time_vec.unsqueeze(0).to(self.device)
        ao, _, _ = self.net.get_action_and_value(mkt, port, sent, tv, det=True)
        return ao.action.squeeze(0).cpu().numpy()


# ==============================================================================
# SECTION 19: Rainbow DQN
# ==============================================================================

class RainbowDQN(nn.Module):
    def __init__(self, cfg: NetworkConfig, nf: int, na: int) -> None:
        super().__init__()
        self.enc = build_encoder(cfg, nf)
        self.dist = DistributionalCritic(
            self.enc.out_dim, na, 51, -10, 10, 128, True)
        self.na = na

    def forward(self, x: torch.Tensor) -> DistributionalOutput:
        f, _ = self.enc(x)
        return self.dist(f)

    def reset_noise(self) -> None:
        if self.training:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()


class RainbowTrainer(BaseTrainer):
    def __init__(self, net: RainbowDQN, cfg: RainbowConfig, dev: torch.device) -> None:
        super().__init__(net, dev)
        self.cfg = cfg
        self.tgt = copy.deepcopy(net).to(dev)
        for p in self.tgt.parameters():
            p.requires_grad_(False)
        self.opt = optim.Adam(
            net.parameters(), lr=cfg.learning_rate, eps=1.5e-4)
        self.replay = PERBuffer(cfg.buffer_size, dev, cfg.per_alpha, cfg.per_beta,
                                cfg.per_beta_frames, cfg.per_epsilon)

    def collect_rollout(self, env: SingleAssetEnv) -> Dict[str, float]:
        self.net.eval()
        res = env.reset()
        er = 0.
        ec = 0
        for _ in range(100):
            x = res.obs_market_seq.unsqueeze(0).to(self.device)
            with torch.no_grad():
                do = self.net(x)
            eps = max(0.01, 1 - self._step / 100_000)
            a = (random.randrange(self.net.na) if random.random() < eps
                 else int(do.q_values.argmax(-1).item()))
            nr = env.step(np.array([a]))
            self.replay.add(Transition(
                res.obs_market_seq.cpu().numpy(), res.obs_portfolio_vec.cpu().numpy(),
                res.obs_sentiment_vec.cpu().numpy(), res.obs_time_vec.cpu().numpy(),
                nr.obs_market_seq.cpu().numpy(),  nr.obs_portfolio_vec.cpu().numpy(),
                nr.obs_sentiment_vec.cpu().numpy(), nr.obs_time_vec.cpu().numpy(),
                np.array([a]), nr.reward, nr.done, nr.info))
            er += nr.reward
            res = nr
            if res.done:
                self._ep_rewards.append(er)
                ec += 1
                er = 0.
                res = env.reset()
        return {"n_episodes": ec, "buffer_size": len(self.replay)}

    def update(self) -> Dict[str, float]:
        if len(self.replay) < self.cfg.batch_size:
            return {"skipped": 1.}
        self.net.train()
        self.net.reset_noise()
        batch, idxs, ws = self.replay.sample(self.cfg.batch_size)
        data, wt = self.replay.collate(batch, ws)
        x = data["market_seqs"]
        nx = data["next_market_seqs"]
        r = data["reward"]
        d = data["done"]
        a = data["actions"].long().squeeze(-1)
        do = self.net(x)
        with torch.no_grad():
            nd = self.tgt(nx)
            na = nd.q_values.argmax(-1)
            nlp = nd.log_probs[torch.arange(
                self.cfg.batch_size, device=self.device), na]
            tgt = self.net.dist.project(r, nlp, d, self.cfg.gamma)
        lp = do.log_probs[torch.arange(
            self.cfg.batch_size, device=self.device), a]
        loss = -(tgt * lp).sum(-1)
        wloss = (loss * wt).mean()
        self.opt.zero_grad()
        wloss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
        self.opt.step()
        self.replay.update_priorities(idxs, loss.detach().cpu().numpy())
        self._step += 1
        if self._step % self.cfg.target_update == 0:
            self.tgt.load_state_dict(self.net.state_dict())
        return {"loss": float(wloss.item()), "q": float(do.q_values.mean().item())}

    @torch.no_grad()
    def _select_action(self, r: StepResult, det: bool) -> np.ndarray:
        self.net.eval()
        x = r.obs_market_seq.unsqueeze(0).to(self.device)
        return np.array([int(self.net(x).q_values.argmax(-1).item())])



def train_on_csv(
    csv_path: str,
    symbol: str = "ASSET",
    timeframe: TimeFrame = TimeFrame.H1,
    n_iterations: int = 100,
    eval_freq: int = 10,
    save_dir: str = "models",
    device_str: str = "auto",
) -> None:
    """Full training pipeline on real CSV data."""

    device = select_device() if device_str == "auto" else torch.device(device_str)
    set_seed(42)

    W = 70
    print("=" * W)
    print(f" ZeroStrike DRL — Real Data Training")
    print("=" * W)
    print(f" Data: {csv_path}")
    print(f" Device: {device}")

    # [1] Load data
    print(f"\n[1/5] Loading data ...")
    bars = load_csv_to_bars(csv_path, symbol=symbol, timeframe=timeframe)
    print(f"      Loaded {len(bars):,} bars")

    # Print data stats
    closes = np.array([b.close for b in bars])
    returns = np.diff(closes) / closes[:-1]
    print(f"      Price: ${closes[0]:.2f} → ${closes[-1]:.2f}")
    print(
        f"      Returns: mean={returns.mean()*100:.3f}% std={returns.std()*100:.3f}%")
    print(
        f"      Sharpe (ann): {(returns.mean()/returns.std()*np.sqrt(252)):.2f}")

    # [2] Split data
    print(f"\n[2/5] Splitting data (70/15/15) ...")
    train_bars, val_bars, test_bars = split_bars_chronological(
        bars, 0.70, 0.15)
    print(f"      Train: {len(train_bars):,} bars")
    print(f"      Val:   {len(val_bars):,} bars")
    print(f"      Test:  {len(test_bars):,} bars")

    # [3] Build network
    print(f"\n[3/5] Building network ...")
    ncfg = NetworkConfig(
        architecture=Architecture.ATTENTION_LSTM,
        d_model=128, n_heads=4, n_layers=2, d_ff=256,
        lstm_hidden=128, lstm_layers=1, lstm_dropout=0.,
        hidden_dims=(256, 128), dropout=0.1,
        use_noisy_net=True, use_dueling=True,
    )
    net = ActorCriticNetwork(ncfg, N_MARKET_FEATURES, N_PORTFOLIO_FEATURES,
                             N_SENTIMENT_FEATURES, N_TIME_FEATURES, ad=3, cont=True).to(device)
    pc = net.param_count()
    print(f"      params: {pc['total']:,}")

    # [4] Setup environments
    print(f"\n[4/5] Setting up environments ...")
    reward_cfg = RewardConfig()

    train_cfg = EnvironmentConfig(
        episode_length=300, warmup_bars=60, initial_capital=100_000.,
        random_start=True, reward=reward_cfg,
    )
    val_cfg = EnvironmentConfig(
        episode_length=300, warmup_bars=60, initial_capital=100_000.,
        random_start=False, reward=reward_cfg,
    )

    train_env = SingleAssetEnv(train_bars, train_cfg, device)
    val_env = SingleAssetEnv(val_bars, val_cfg, device)

    pcfg = PPOConfig(
        n_steps=512, n_epochs=4, batch_size=64,
        learning_rate=3e-4, warmup_steps=128,
        target_kl=0.05,
        normalize_advantage=True, normalize_returns=True,
        entropy_coef=0.01, entropy_coef_min=0.001, entropy_anneal_steps=100_000,
        clip_epsilon=0.2, value_coef=0.5, max_grad_norm=0.5,
    )
    trainer = PPOTrainer(net, pcfg, device)

    # [5] Train
    print(f"\n[5/5] Training ({n_iterations} iterations) ...")
    print(f"  {'it':>4}  {'train_R':>8}  {'val_R':>8}  {'capital':>10}  {'trades':>7}  {'WR':>6}  {'KL':>7}  {'GN':>6}")
    print(f"  {'─'*75}")

    best_val = -float('inf')
    best_capital = 100_000.0
    os.makedirs(save_dir, exist_ok=True)

    for i in range(n_iterations):
        rm = trainer.collect_rollout(train_env)
        um = trainer.update()

        if i % eval_freq == 0 or i == n_iterations - 1:
            em = trainer.evaluate(val_env, n=3, det=True)
            val_r = em['mean_reward']
            val_cap = em['mean_capital']
            val_wr = em.get('mean_win_rate', 0)
            val_tr = em.get('mean_trades', 0)

            status = ""
            if val_r > best_val:
                best_val = val_r
                best_capital = val_cap
                trainer.save(os.path.join(save_dir, "best_model.pt"))
                status = "↑ BEST"

            print(f"  {i:>4}  {rm['mean_reward']:>+8.2f}  {val_r:>+8.2f}  {val_cap:>10,.0f}  {val_tr:>7.0f}  {val_wr:>5.1%}  {um['approx_kl']:>7.5f}  {um['grad_norm']:>6.2f}  {status}")

    # Final test evaluation
    print(f"\n{'='*W}")
    print(f" Final Test Evaluation")
    print(f"{'='*W}")

    test_cfg = EnvironmentConfig(
        episode_length=500, warmup_bars=60, initial_capital=100_000.,
        random_start=False, reward=reward_cfg,
    )
    test_env = SingleAssetEnv(test_bars, test_cfg, device)

    # Load best model
    trainer.load(os.path.join(save_dir, "best_model.pt"))

    em = trainer.evaluate(test_env, n=5, det=True)
    print(f"  Reward:    {em['mean_reward']:+.2f} ± {em['std_reward']:.2f}")
    print(f"  Capital:   ${em['mean_capital']:,.0f}")
    print(f"  Win Rate:  {em.get('mean_win_rate', 0):.1%}")
    print(f"  Trades:    {em.get('mean_trades', 0):.0f}")

    # Backtest
    bt = Backtester(trainer, device)
    metrics, _ = bt.run(test_env, seed=0, verbose=False)
    print(f"\n  Backtest:")
    print(f"  Sharpe:    {metrics.sharpe_ratio:.3f}")
    print(f"  Max DD:    {metrics.max_drawdown:.2%}")
    print(f"  Return:    {metrics.total_return:+.2%}")
    print(f"  Win Rate:  {metrics.win_rate:.1%}")
    print(f"  Trades:    {metrics.total_trades}")

    print(f"\n{'='*W}")
    print(f" ✓ Training complete. Model saved to {save_dir}/best_model.pt")
    print(f"{'='*W}")


if __name__ == "__main__":
    import sys

    # Default to BTC hourly data
    csv_path = "market_data/BTC-USD_1h_2024-05-31_to_2026-05-29.csv"

    # Allow command-line override
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    # Detect symbol and timeframe from filename
    filename = os.path.basename(csv_path)
    if "_1h_" in filename:
        timeframe = TimeFrame.H1
        symbol = filename.split("_1h_")[0]
    elif "_1d_" in filename:
        timeframe = TimeFrame.D1
        symbol = filename.split("_1d_")[0]
    else:
        timeframe = TimeFrame.H1
        symbol = "ASSET"

    train_on_csv(
        csv_path=csv_path,
        symbol=symbol,
        timeframe=timeframe,
        n_iterations=100,
        eval_freq=5,
    )
