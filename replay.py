
from __future__ import annotations


import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch import Tensor


@dataclass
class Transition:
    mkt:   np.ndarray
    port:  np.ndarray
    sent:  np.ndarray
    tv:    np.ndarray
    nmkt:  np.ndarray
    nport: np.ndarray
    nsent: np.ndarray
    ntv:   np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


class SumTree:
    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.tree = np.zeros(2 * cap - 1, np.float64)
        self.data: List[Any] = [None] * cap
        self._ptr = 0
        self._sz = 0

    @property
    def total(self) -> float: return float(self.tree[0])

    @property
    def size(self) -> int: return self._sz

    def add(self, p: float, d: Any) -> None:
        idx = self._ptr + self.cap - 1
        self.data[self._ptr] = d
        self.update(idx, p)
        self._ptr = (self._ptr + 1) % self.cap
        self._sz = min(self._sz + 1, self.cap)

    def update(self, i: int, p: float) -> None:
        delta = p - self.tree[i]
        self.tree[i] = p
        while i > 0:
            i = (i - 1) // 2
            self.tree[i] += delta

    def get(self, s: float) -> Tuple[int, float, Any]:
        i = 0
        while True:
            left = 2 * i + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left] + 1e-12:
                i = left
            else:
                s -= self.tree[left]
                i = left + 1
        di = i - self.cap + 1
        if di < 0 or di >= self._sz or self.data[di] is None:
            di = random.randrange(max(self._sz, 1))
            i = di + self.cap - 1
        return i, float(self.tree[i]), self.data[di]

    def min_p(self) -> float:
        if self._sz == 0:
            return 1.
        lv = self.tree[self.cap - 1: self.cap - 1 + self._sz]
        pos = lv[lv > 0]
        return float(pos.min()) if len(pos) else 1.


class UniformBuf:
    def __init__(self, cap: int, dev: torch.device) -> None:
        self.cap = cap
        self.dev = dev
        self._b: List[Optional[Transition]] = [None] * cap
        self._ptr = 0
        self._sz = 0

    def add(self, t: Transition) -> None:
        self._b[self._ptr] = t
        self._ptr = (self._ptr + 1) % self.cap
        self._sz = min(self._sz + 1, self.cap)

    def sample(self, n: int) -> List[Transition]:
        return [self._b[i] for i in random.sample(range(self._sz), min(n, self._sz))]

    def collate(self, batch: List[Transition]) -> Dict[str, Tensor]:
        def _t(a): return torch.from_numpy(np.stack(a)).float().to(self.dev)
        return {"market_seqs":         _t([t.mkt for t in batch]),
                "portfolio_vecs":      _t([t.port for t in batch]),
                "sentiment_vecs":      _t([t.sent for t in batch]),
                "time_vecs":           _t([t.tv for t in batch]),
                "next_market_seqs":    _t([t.nmkt for t in batch]),
                "next_portfolio_vecs": _t([t.nport for t in batch]),
                "next_sentiment_vecs": _t([t.nsent for t in batch]),
                "next_time_vecs":      _t([t.ntv for t in batch]),
                "actions": _t([t.action for t in batch]),
                "reward":  torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.dev),
                "done":    torch.tensor([float(t.done) for t in batch], dtype=torch.float32, device=self.dev)}

    def __len__(self) -> int: return self._sz
    def is_ready(self, n: int) -> bool: return self._sz >= n


class PERBuffer:
    def __init__(self, cap: int, dev: torch.device,
                 alpha: float = 0.6, beta0: float = 0.4,
                 beta_f: int = 100_000, eps: float = 1e-6) -> None:
        self.dev = dev
        self.alpha = alpha
        self.beta0 = beta0
        self.beta_f = beta_f
        self.eps = eps
        self.tree = SumTree(cap)
        self._maxp = 1.
        self._frame = 1
        self._h = UniformBuf(1, dev)

    @property
    def beta(self) -> float:
        return min(1., self.beta0 + (1 - self.beta0) * self._frame / self.beta_f)

    def add(self, t: Transition, p: Optional[float] = None) -> None:
        self.tree.add((p if p is not None else self._maxp) ** self.alpha, t)

    def sample(self, n: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        self._frame += 1
        trans: List[Transition] = []
        idxs = np.zeros(n, np.int32)
        ws = np.zeros(n, np.float32)
        seg = self.tree.total / n
        beta = self.beta
        minp = self.tree.min_p() / (self.tree.total + 1e-10)
        maxw = (self.tree.size * minp + 1e-10) ** (-beta)
        for i in range(n):
            s = random.uniform(seg * i, seg * (i + 1))
            idx, p, d = self.tree.get(max(s, 1e-12))
            idxs[i] = idx
            prob = p / (self.tree.total + 1e-10)
            ws[i] = (self.tree.size * prob + 1e-10) ** (-beta) / (maxw + 1e-10)
            trans.append(d)
        return trans, idxs, ws.astype(np.float32)

    def update_priorities(self, idxs: np.ndarray, td: np.ndarray) -> None:
        ps = (np.abs(td) + self.eps) ** self.alpha
        self._maxp = max(self._maxp, float(ps.max()))
        for i, p in zip(idxs.tolist(), ps.tolist()):
            self.tree.update(int(i), float(p))

    def collate(self, batch: List[Transition], ws: np.ndarray) -> Tuple[Dict, Tensor]:
        return self._h.collate(batch), torch.tensor(ws, dtype=torch.float32, device=self.dev)

    def __len__(self) -> int: return self.tree.size
    def is_ready(self, n: int) -> bool: return self.tree.size >= n

