# zerostrike/drl/state/normaliser.py

from __future__ import annotations
import torch
import torch.nn as nn 
from torch import Tensor


class WelfordNorm(nn.Module):
    """
    Online Welford normalisation implemented entirely in PyTorch.

    Keeps running mean and variance without storing the full dataset.
    Numerically stable (Welford's algorithm).
    All tensors on the same device — no CPU↔GPU transfers.
    """

    def __init__(self, n: int, device: torch.device) -> None:
        super().__init__()
        self.register_buffer("_cnt",  torch.tensor(
            0, dtype=torch.int64,   device=device))
        self.register_buffer("_mean", torch.zeros(
            n,  dtype=torch.float64, device=device))
        self.register_buffer("_M2",   torch.zeros(
            n,  dtype=torch.float64, device=device))

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        x = x.to(dtype=torch.float64, device=self._mean.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        B = x.size(0)
        bc = torch.tensor(B, dtype=torch.int64, device=x.device)
        tot = self._cnt + bc
        bm = x.mean(0)
        bM = ((x - bm) ** 2).sum(0)
        delta = bm - self._mean
        self._mean = self._mean + delta * bc.double() / tot.clamp(min=1).double()
        self._M2 = self._M2 + bM + delta ** 2 * \
            (self._cnt * bc).double() / tot.clamp(min=1).double()
        self._cnt = tot

    @torch.no_grad()
    def normalise(self, x: Tensor, clip: float = 5.0) -> Tensor:
        if self._cnt < 2:
            return x
        var = self._M2 / (self._cnt - 1).clamp(min=1).double()
        std = (var + 1e-8).sqrt().to(torch.float32)
        mean = self._mean.to(torch.float32)
        return ((x - mean) / (std + 1e-8)).clamp(-clip, clip)

    def reset_stats(self) -> None:
        self._cnt.zero_()
        self._mean.zero_()
        self._M2.zero_()
