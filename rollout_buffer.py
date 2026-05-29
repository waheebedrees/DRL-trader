
from __future__ import annotations

from typing import Dict, Optional
import torch
from torch import Tensor

from state import StepResult


class RolloutBuf:
    def __init__(self, dev: torch.device) -> None:
        self.dev = dev
        self.clear()

    def add(self, r: StepResult, a: Tensor, lp: Tensor, v: Tensor,
            reward: float, done: float) -> None:
        self._m.append(r.obs_market_seq.detach().cpu())
        self._p.append(r.obs_portfolio_vec.detach().cpu())
        self._s.append(r.obs_sentiment_vec.detach().cpu())
        self._t.append(r.obs_time_vec.detach().cpu())
        self._a.append(a.detach().cpu())
        self._lp.append(lp.detach().cpu())
        self._v.append(v.detach().cpu())
        self._r.append(float(reward))
        self._d.append(float(done))

    def clear(self) -> None:
        self._m = []
        self._p = []
        self._s = []
        self._t = []
        self._a = []
        self._lp = []
        self._v = []
        self._r = []
        self._d = []
        self.adv: Optional[Tensor] = None
        self.ret: Optional[Tensor] = None

    def __len__(self) -> int: return len(self._r)

    def tensors(self) -> Dict[str, Tensor]:
        return {"mkt":  torch.stack(self._m).to(self.dev),
                "port": torch.stack(self._p).to(self.dev),
                "sent": torch.stack(self._s).to(self.dev),
                "tv":   torch.stack(self._t).to(self.dev),
                "act":  torch.stack(self._a).to(self.dev),
                "lp":   torch.stack(self._lp).to(self.dev),
                "val":  torch.stack(self._v).squeeze(-1).to(self.dev),
                "rew":  torch.tensor(self._r, dtype=torch.float32, device=self.dev),
                "don":  torch.tensor(self._d, dtype=torch.float32, device=self.dev)}

    @torch.no_grad()
    def gae(self, T: Dict, lv: Tensor, last_done: bool,
            gamma: float, lam: float, norm_adv: bool, norm_ret: bool) -> None:
        N = T["rew"].size(0)
        val = T["val"]
        vals_next = torch.empty_like(val)
        vals_next[:-1] = val[1:]
        vals_next[-1] = 0. if last_done else lv.squeeze().item()
        adv = torch.zeros(N, device=self.dev)
        g = 0.
        for t in reversed(range(N)):
            nt = 1. - T["don"][t].item()
            delta = T["rew"][t].item() + gamma * \
                vals_next[t].item() * nt - val[t].item()
            g = delta + gamma * lam * nt * g
            adv[t] = g
        ret = adv + val
        if norm_ret:
            ret = (ret - ret.mean()) / (ret.std().clamp(1e-8))
        if norm_adv:
            adv = (adv - adv.mean()) / (adv.std().clamp(1e-8))
        self.adv = adv
        self.ret = ret

    def batches(self, T: Dict, bs: int):
        N = T["rew"].size(0)
        for idx in torch.randperm(N).split(bs):
            if not len(idx):
                continue
            idx = idx.to(self.dev)
            yield {"mkt":  T["mkt"][idx],  "port": T["port"][idx],
                   "sent": T["sent"][idx], "tv":   T["tv"][idx],
                   "act":  T["act"][idx],  "olp":  T["lp"][idx],
                   "oval": T["val"][idx],  "adv":  self.adv[idx], "ret": self.ret[idx]}
