
"""
DRL Utilities
========================
Diagnostic tools, data generators, validation utilities,
and PyTorch-specific helpers for the DRL trading framework v3.6.

Usage:
    from utils import (
        Diagnostics, DataGenerator, WalkForwardValidator,
        GradientMonitor, ModelAnalyzer, MemoryOptimizer
    )
"""




from .diagnostics import Diagnostics, RewardAnalyzer, PolicyMonitor
from .data_generator import DataGenerator, RealisticMarketSimulator
from .validation import WalkForwardValidator, CrossValidator
from .visualization import Dashboard, plot_training_curves, plot_reward_decomposition
from .callbacks import EarlyStopping, ModelCheckpoint, WandBLogger

# PyTorch utilities
from .torch_utils import (
    init_network,
    init_lstm_forget_gates,
    init_layer_scale,
    GradientMonitor,
    GradientStats,
    MemoryOptimizer,
    ModelAnalyzer,
    LayerInfo,
    LayerProfiler,
    freeze_layers,
    unfreeze_layers,
    get_device_info,
    print_device_info,
    set_optimizer_parameters,
    analyze_weights,
    diagnose_training,
    print_diagnosis,
)



__version__ = "1.0.0"
__all__ = [

    # Diagnostics
    'Diagnostics', 'RewardAnalyzer', 'PolicyMonitor',

    # Data
    'DataGenerator', 'RealisticMarketSimulator',

    # Validation
    'WalkForwardValidator', 'CrossValidator',

    # Visualization
    'Dashboard', 'plot_training_curves', 'plot_reward_decomposition',

    # Callbacks
    'EarlyStopping', 'ModelCheckpoint', 'WandBLogger',

    # PyTorch Utils - Initialization
    'init_network', 'init_lstm_forget_gates', 'init_layer_scale',

    # PyTorch Utils - Monitoring
    'GradientMonitor', 'GradientStats', 'LayerProfiler',

    # PyTorch Utils - Optimization
    'MemoryOptimizer', 'set_optimizer_parameters',

    # PyTorch Utils - Analysis
    'ModelAnalyzer', 'LayerInfo', 'analyze_weights',

    # PyTorch Utils - Helpers
    'freeze_layers', 'unfreeze_layers',
    'get_device_info', 'print_device_info',
    'diagnose_training', 'print_diagnosis',
]
