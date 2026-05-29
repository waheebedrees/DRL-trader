

from __future__ import annotations

from abc import ABC, abstractmethod
import os
import random
import numpy as np
from dataclasses import dataclass
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

    MarketConfig,
    EnvironmentNotResetError,
    EpisodeTerminatedError,
    InsufficientDataError,
    TrainingError,
    TrainingConfig,
    CheckpointError,
    ConfigurationError,

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

from execution_simulator import ExecutionConfig, ExecutionSimulator
from reward_engine import RewardEngine
from networks.actor_critic_network import ActorCriticNetwork
from rollout_buffer import RolloutBuf

from single_asset_env import SingleAssetEnv
# ==============================================================================
# SECTION 15: Base Trainer
# ==============================================================================

class BaseTrainer(ABC):
    def __init__(self, net: nn.Module, dev: torch.device) -> None:
        self.net = net.to(dev)
        self.device = dev
        self._step = 0
        self._ep_rewards: List[float] = []

    @property
    def global_step(self) -> int: return self._step

    @abstractmethod
    def collect_rollout(self, env: SingleAssetEnv) -> Dict[str, float]: ...
    @abstractmethod
    def update(self) -> Dict[str, float]: ...
    @abstractmethod
    def _select_action(self, r: StepResult, det: bool) -> np.ndarray: ...

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"net": self.net.state_dict(),
                   "step": self._step, **self._extra_save()}, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ck["net"])
        self._step = ck.get("step", 0)
        self._extra_load(ck)

    def _extra_save(self) -> Dict: return {}
    def _extra_load(self, ck: Dict) -> None: pass

    @torch.no_grad()
    def evaluate(self, env: SingleAssetEnv, n: int = 5, det: bool = True) -> Dict[str, float]:
        self.net.eval()
        rews, caps, trades, wrs = [], [], [], []
        for ep in range(n):
            r = env.reset(seed=ep)
            er = 0.
            while not env.is_done:
                r = env.step(self._select_action(r, det))
                er += r.reward
            s = env.episode_stats
            rews.append(er)
            caps.append(s.final_capital)
            trades.append(s.total_trades)
            wrs.append(s.win_rate)
        return {"mean_reward":   float(np.mean(rews)),   "std_reward":    float(np.std(rews)),
                "mean_capital":  float(np.mean(caps)),   "mean_trades":   float(np.mean(trades)),
                "mean_win_rate": float(np.mean(wrs))}

