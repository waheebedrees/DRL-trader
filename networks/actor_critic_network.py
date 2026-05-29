

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor


from config import NetworkConfig


from networks import (
    # Layers
    NoisyLinear,
    GatedResidualBlock,
    PortfolioEncoder,
    build_encoder,
    # Heads
    ContinuousActor,
    DiscreteActor,
    DuelingCritic,

    ActorOutput,
    CriticOutput,
)


class ActorCriticNetwork(nn.Module):
    """
    Full Actor-Critic network for PPO / SAC.

    Architecture:
    ┌─────────────────────────────────────────────────────┐
    │  Market Seq [B,T,F]  ─→  MarketEncoder  ─→ [B, D]  │
    │  Portfolio  [B,P]    ─→  PortfolioEnc   ─→ [B, D/4]│
    │  Sentiment  [B,S]    ─→  Linear          ─→ [B, 32] │
    │  Time       [B,6]    ─→  Linear          ─→ [B, 16] │
    │                          Concatenate                  │
    │                          ↓                           │
    │                     Shared Trunk                     │
    │                   (MLP + GRN blocks)                 │
    │                    /              \\                  │
    │               Actor              Critic              │
    │          (Gaussian dist)     (Dueling Q)             │
    └─────────────────────────────────────────────────────┘
    """

    def __init__(self, cfg: NetworkConfig, nm: int, np_: int,
                 ns: int = 5, nt: int = 6, ad: int = 3, cont: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_dim = ad
        self.continuous = cont
        D = cfg.d_model
        self.menc = build_encoder(cfg, nm)
        md = self.menc.out_dim
        pd = max(D // 4, 16)
        self.penc = PortfolioEncoder(np_, pd, cfg.dropout)
        self.senc = nn.Sequential(
            nn.Linear(ns, 32), nn.LayerNorm(32), nn.GELU())
        self.tenc = nn.Sequential(nn.Linear(nt, 16), nn.GELU())
        dims = [md + pd + 32 + 16] + list(cfg.hidden_dims)
        trunk: List[nn.Module] = []
        for i in range(len(dims) - 1):
            trunk += [nn.Linear(dims[i], dims[i + 1]), nn.LayerNorm(dims[i + 1]),
                      nn.GELU(), nn.Dropout(cfg.dropout)]
            if i < len(dims) - 2:
                trunk.append(GatedResidualBlock(
                    dims[i + 1], dims[i + 1], cfg.dropout))
        self.trunk = nn.Sequential(*trunk)
        fd = cfg.hidden_dims[-1]
        self.actor = (ContinuousActor(fd, ad, (fd // 2,), cfg.dropout) if cont
                      else DiscreteActor(fd, ad, (fd // 2,)))
        self.critic = DuelingCritic(fd, ad, fd // 2, cfg.use_noisy_net)
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear) and not isinstance(m, NoisyLinear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.actor.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 0.01)

    def enc_params(self) -> List[nn.Parameter]:
        return (list(self.menc.parameters()) + list(self.penc.parameters())
                + list(self.senc.parameters()) + list(self.tenc.parameters()))

    def non_enc_params(self) -> List[nn.Parameter]:
        ep = {id(p) for p in self.enc_params()}
        return [p for p in self.parameters() if id(p) not in ep]

    def encode(self, mkt: Tensor, port: Tensor,
               sent: Optional[Tensor] = None, tv: Optional[Tensor] = None,
               hx: Optional[Tuple] = None) -> Tuple[Tensor, Optional[Tuple]]:
        enc = self.menc(mkt, hx)
        mo, hx_out = (enc if isinstance(enc, tuple) else (enc, None))
        po = self.penc(port)
        B = mkt.size(0)
        so = self.senc(sent) if sent is not None else torch.zeros(
            B, 32, device=mkt.device)
        to = self.tenc(tv) if tv is not None else torch.zeros(
            B, 16, device=mkt.device)
        return self.trunk(torch.cat([mo, po, so, to], -1)), hx_out

    def get_value(self, mkt: Tensor, port: Tensor,
                  sent: Optional[Tensor] = None, tv: Optional[Tensor] = None) -> Tensor:
        f, _ = self.encode(mkt, port, sent, tv)
        return self.critic.get_value(f)

    def get_action_and_value(
            self, mkt: Tensor, port: Tensor,
            sent: Optional[Tensor] = None, tv: Optional[Tensor] = None,
            action: Optional[Tensor] = None, hx: Optional[Tuple] = None, det: bool = False,
    ) -> Tuple[ActorOutput, CriticOutput, Optional[Tuple]]:
        f, hx_out = self.encode(mkt, port, sent, tv, hx)
        return self.actor(f, action, det), CriticOutput(self.critic.get_value(f)), hx_out

    def reset_noise(self) -> None:
        if self.training:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()

    def param_count(self) -> Dict[str, int]:
        return {"total":     sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)}
