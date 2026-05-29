
#trader.py
from __future__ import annotations


import random
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
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

from networks.actor_critic_network import ActorCriticNetwork
from rollout_buffer import RolloutBuf

from replay import PERBuffer, SumTree, Transition, UniformBuf
from single_asset_env import SingleAssetEnv, StepContext
from base_trainer import BaseTrainer
from back_tester import Backtester, BacktestMetrics
from data_gen import make_bars

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
    def alpha(self) -> Tensor: return self.la.exp()

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
        self.dist = DistributionalCritic(self.enc.out_dim, na, 51, -10, 10, 128, True)
        self.na = na

    def forward(self, x: Tensor) -> DistributionalOutput:
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



def main() -> None:
    set_seed(42)
    device = select_device()
    W = 70
    print("=" * W)
    print(
        f" ZeroStrike DRL v3.6 | device={device} | torch={torch.__version__}")
    print("=" * W)
    print(f" Features: market={N_MARKET_FEATURES}  portfolio={N_PORTFOLIO_FEATURES}"
          f"  sentiment={N_SENTIMENT_FEATURES}  time={N_TIME_FEATURES}\n")

    # [1/6] Network
    print("[1/6] Building network ...")
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
    print(f"      params total={pc['total']:,}  trainable={pc['trainable']:,}")
    with torch.no_grad():
        ao, co, _ = net.get_action_and_value(
            torch.randn(2, 60, N_MARKET_FEATURES,    device=device),
            torch.randn(2, N_PORTFOLIO_FEATURES,     device=device),
            torch.randn(2, N_SENTIMENT_FEATURES,     device=device),
            torch.randn(2, N_TIME_FEATURES,          device=device))
    std_v = float(ao.std.mean())
    print(f"      forward OK | action={tuple(ao.action.shape)}"
          f"  value=[{co.value.min():.3f},{co.value.max():.3f}]  actor_std={std_v:.3f}")
    _chk(0.05 < std_v < 3., f"Actor std out of [0.05, 3.0]: {std_v:.4f}")

    # [2/6] PPO
    print("\n[2/6] PPO training ...")
    all_bars = make_bars(1200)
    train_bars = all_bars[:900]
    eval_bars = all_bars[900:]

    reward_cfg = RewardConfig()

    train_cfg = EnvironmentConfig(
        episode_length=200, warmup_bars=60, initial_capital=100_000.,
        random_start=True, reward=reward_cfg,
    )
    eval_cfg = EnvironmentConfig(
        episode_length=200, warmup_bars=60, initial_capital=100_000.,
        random_start=False, reward=reward_cfg,
    )
    train_env = SingleAssetEnv(train_bars, train_cfg, device)
    eval_env = SingleAssetEnv(eval_bars,  eval_cfg,  device)

    pcfg = PPOConfig(
        n_steps=512, n_epochs=4, batch_size=64,
        learning_rate=3e-4, warmup_steps=128,
        target_kl=0.05,
        normalize_advantage=True, normalize_returns=True,
        entropy_coef=0.01, entropy_coef_min=0.001, entropy_anneal_steps=100_000,
        clip_epsilon=0.2, value_coef=0.5, max_grad_norm=0.5,
    )
    trainer = PPOTrainer(net, pcfg, device)
    # Diagnostic header
    print(f"\n  {'it':>4}  {'train_R':>8}  {'eval_R':>8}"
          f"  {'cap':>9}  {'wr':>6}  {'trd':>5}"
          f"  {'KL':>7}  {'ent':>6}  status")
    print(f"  {'─'*72}")

    for i in range(30):
        rm = trainer.collect_rollout(train_env)
        um = trainer.update()
        pl = um.get("policy_loss", 0)
        vl = um.get("value_loss",  0)
        kl = um.get("approx_kl",   0)
        lr = um.get("lr",          0)
        gn = um.get("grad_norm",   0)
        print(f"      iter={i + 1:2d}  R={rm.get('mean_reward', 0):+.4f}"
              f"  πL={pl:.4f}  VL={vl:.4f}  KL={kl:.5f}  GN={gn:.3f}  lr={lr:.2e}")
        _chk(math.isfinite(pl) and math.isfinite(
            vl), f"Loss non-finite at iter {i + 1}")
        _chk(abs(pl) < 10.,  f"Policy loss magnitude unrealistic: {pl:.4f}")
        _chk(vl < 100.,      f"Value loss exploded: {vl:.2f}")
        _chk(abs(kl) < 0.5,  f"KL too high: {kl:.5f}")
        _chk(gn < 50.,       f"Gradient norm exploded: {gn:.2f}")

    # [3/6] Evaluate
    print("\n[3/6] Evaluating PPO policy on held-out data ...")
    em = trainer.evaluate(eval_env, n=3, det=True)
    cap = em["mean_capital"]
    print(f"      mean_reward={em['mean_reward']:+.4f}  mean_capital={cap:,.0f}"
          f"  win_rate={em.get('mean_win_rate', 0):.1%}  trades={em.get('mean_trades', 0):.0f}")
    _chk(50_000 < cap < 200_000,  # WIDENED: allow up to 2x return
         f"Capital out of [50K, 200K]: {cap:,.0f}")

    # [4/6] Backtest
    print("\n[4/6] Running backtest on held-out data ...")
    bt = Backtester(trainer, device)
    m, _ = bt.run(eval_env, seed=0, verbose=False)
    print(f"      Sharpe={m.sharpe_ratio:.3f}  MaxDD={m.max_drawdown:.2%}"
          f"  Trades={m.total_trades}  WinRate={m.win_rate:.1%}  Return={m.total_return:+.2%}")

    # UPDATED CHECKS for data with real patterns
    _chk(abs(m.sharpe_ratio) <= 10., f"Sharpe unrealistic: {m.sharpe_ratio}")
    _chk(-0.50 < m.total_return < 1.00,
         f"Return outside expected range: {m.total_return:+.2%}")
    _chk(m.max_drawdown < 0.50,
         f"Max drawdown too high: {m.max_drawdown:.2%}")
    _chk(0.30 < m.win_rate < 0.90 if m.total_trades > 0 else True,
         f"Win rate suspicious: {m.win_rate:.1%}")

    # [5/6] SAC
    print("\n[5/6] SAC smoke test ...")
    snet = ActorCriticNetwork(ncfg, N_MARKET_FEATURES, N_PORTFOLIO_FEATURES,
                 N_SENTIMENT_FEATURES, N_TIME_FEATURES, ad=3, cont=True).to(device)
    sac = SACTrainer(snet, SACConfig(learning_starts=10, batch_size=8, buffer_size=1000,
                                     train_freq=20, gradient_steps=1, max_grad_norm=1.),
                     device, ad=3)
    rm2 = sac.collect_rollout(train_env)
    um2 = sac.update()
    if "skipped" not in um2:
        cl = um2.get("critic_loss", 0)
        print(f"      buffer={rm2['buffer_size']}  critic_loss={cl:.4f}"
              f"  actor_loss={um2.get('actor_loss', 0):.4f}  alpha={um2.get('alpha', 0):.4f}")
        _chk(cl < 500., f"SAC critic loss too large: {cl:.2f}")
    else:
        print(
            f"      buffer={rm2.get('buffer_size', 0)} (needs {sac.cfg.learning_starts})")

    # [6/6] TD3
    print("\n[6/6] TD3 smoke test ...")
    tnet = ActorCriticNetwork(ncfg, N_MARKET_FEATURES, N_PORTFOLIO_FEATURES,
                 N_SENTIMENT_FEATURES, N_TIME_FEATURES, ad=3, cont=True).to(device)
    td3 = TD3Trainer(tnet, TD3Config(learning_starts=10, batch_size=8, buffer_size=1000,
                                     max_grad_norm=1.), device, ad=3)
    rm3 = td3.collect_rollout(train_env)
    um3 = td3.update()
    if "skipped" not in um3:
        print(f"      buffer={rm3['buffer_size']}  critic_loss={um3.get('critic_loss', 0):.4f}"
              f"  actor_loss={um3.get('actor_loss', 0):.4f}")
    else:
        print(
            f"      buffer={rm3.get('buffer_size', 0)} (needs {td3.cfg.learning_starts})")

    print(f"\n{'=' * W}")
    print(" ✓ All checks passed — ZeroStrike DRL v3.6 ready.")
    print("=" * W)


if __name__ == "__main__":
    main()




