"""
ZeroStrike Networks Package
============================
Clean, modular neural network components for DRL trading.

Usage:
    from networks import (
        NoisyLinear, GatedResidualBlock, PositionalEncoding,
        RotaryEmbedding, MarketAttention, SwiGLUFFN,
        DilatedCausalConv, SqueezeExcitation,
        MarketTransformerEncoder, AttentionLSTMEncoder,
        TemporalCNNEncoder, HybridEncoder, PortfolioEncoder,
        ContinuousActor, DuelingCritic, DistributionalCritic,
        build_encoder
    )
"""


from .layers import (
    NoisyLinear,
    PositionalEncoding,
    RotaryEmbedding,
    MarketAttention,
    GatedResidualBlock,
    SqueezeExcitation,
    DilatedCausalConv,
    SwiGLUFFN,
    
)
from .encoders import (

    MarketTransformerEncoder,
    AttentionLSTMEncoder,
    TemporalCNNEncoder,
    HybridEncoder,
    PortfolioEncoder,
    ActorOutput,
    CriticOutput,
    DistributionalOutput,
    ContinuousActor,
    DuelingCritic,
    DistributionalCritic,
    DiscreteActor,
    TwinQCritic,
    build_encoder,
)

__all__ = [
    # Layers
    'NoisyLinear', 'GatedResidualBlock', 'PositionalEncoding',
    'RotaryEmbedding', 'MarketAttention', 'SwiGLUFFN',
    'DilatedCausalConv', 'SqueezeExcitation',
    # Encoders
    'MarketTransformerEncoder', 'AttentionLSTMEncoder',
    'TemporalCNNEncoder', 'HybridEncoder', 'PortfolioEncoder',
    'build_encoder',
    # Heads
    'ContinuousActor', 'DuelingCritic', 'DistributionalCritic',   "DiscreteActor", "TwinQCritic",
    # Types
    'ActorOutput', 'CriticOutput', 'DistributionalOutput',
]
