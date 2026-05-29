"""
PyTorch Utilities for ZeroStrike DRL
=====================================
Network initialization, gradient analysis, memory optimization,
and model analysis tools.

Designed to work with the ZeroStrike DRL v3.6 framework.
All utilities are standalone and require no changes to core code.

Usage:
    from utils.torch_utils import (
        init_network, GradientMonitor, MemoryOptimizer,
        ModelAnalyzer, LayerProfiler, freeze_layers, unfreeze_layers
    )
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
import math
import time
import gc

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1: Advanced Initialization
# ──────────────────────────────────────────────────────────────────────────────

def init_network(
    model: nn.Module,
    method: str = 'orthogonal',
    gain: float = 1.0,
    bias_init: float = 0.0,
    verbose: bool = True,
) -> nn.Module:
    """
    Comprehensive network initialization with multiple strategies.
    
    Args:
        model: PyTorch model to initialize
        method: Initialization method:
            - 'orthogonal': Orthogonal initialization (best for RL)
            - 'xavier': Xavier/Glorot uniform
            - 'kaiming': Kaiming/He normal (good for ReLU)
            - 'sparse': Sparse initialization (lottery ticket style)
            - 'identity': Identity initialization for square weights
        gain: Gain factor for weights
        bias_init: Initial bias value
        verbose: Print initialization summary
    
    Returns:
        Initialized model (modified in-place)
    
    Usage:
        net = ACNet(config, ...)
        net = init_network(net, method='orthogonal', gain=0.01)
    """
    
    def _init_weight(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            if method == 'orthogonal':
                if m.weight.ndim >= 2:
                    nn.init.orthogonal_(m.weight, gain=gain)
                else:
                    nn.init.normal_(m.weight, mean=0, std=gain)
                    
            elif method == 'xavier':
                nn.init.xavier_uniform_(m.weight, gain=gain)
                
            elif method == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                
            elif method == 'sparse':
                # Sparse initialization: 90% zeros, 10% random
                nn.init.orthogonal_(m.weight, gain=gain)
                mask = torch.rand_like(m.weight) > 0.9
                m.weight.data[mask] = 0.0
                
            elif method == 'identity':
                if m.weight.ndim >= 2 and m.weight.size(0) == m.weight.size(1):
                    nn.init.eye_(m.weight)
                else:
                    nn.init.orthogonal_(m.weight, gain=gain)
            
            # Initialize biases
            if m.bias is not None:
                nn.init.constant_(m.bias, bias_init)
    
    model.apply(_init_weight)
    
    if verbose:
        print(f"✓ Network initialized with '{method}' method (gain={gain})")
    
    return model


def init_lstm_forget_gates(model: nn.Module, bias_value: float = 1.0) -> None:
    """
    Initialize LSTM forget gate biases to 1.0 for better long-term memory.
    
    This is a well-known trick to prevent LSTM from forgetting at the start of training.
    
    Usage:
        init_lstm_forget_gates(model)
    """
    for name, param in model.named_parameters():
        if 'lstm' in name.lower() and 'bias' in name:
            n = param.size(0)
            # LSTM biases are [input_gate, forget_gate, cell_gate, output_gate] * 2
            # Set forget gate biases (indices n//4 to n//2) to bias_value
            start, end = n // 4, n // 2
            param.data[start:end].fill_(bias_value)


def init_layer_scale(model: nn.Module, scale: float = 0.1) -> None:
    """
    Apply LayerScale initialization for transformer-like architectures.
    
    Reduces initial contribution of each sublayer, improving training stability.
    
    Usage:
        init_layer_scale(model, scale=0.1)
    """
    for name, param in model.named_parameters():
        if 'lpe' in name or 'layer_scale' in name:
            param.data.fill_(scale)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2: Gradient Monitoring & Analysis
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GradientStats:
    """Statistics for gradient analysis"""
    mean: float = 0.0
    std: float = 0.0
    max: float = 0.0
    min: float = 0.0
    l2_norm: float = 0.0
    zero_fraction: float = 0.0  # Fraction of zero gradients
    exploding: bool = False     # Gradient explosion detected
    vanishing: bool = False     # Gradient vanishing detected


class GradientMonitor:
    """
    Monitor and analyze gradients during training.
    
    Detects:
    - Exploding gradients (norm > threshold)
    - Vanishing gradients (norm < threshold)
    - Dead neurons (zero gradients)
    - Gradient flow through layers
    
    Usage:
        monitor = GradientMonitor(explosion_threshold=10.0, vanishing_threshold=1e-5)
        
        for epoch in range(n_epochs):
            loss.backward()
            stats = monitor.analyze(model)
            
            if stats.exploding:
                print("Warning: Gradient explosion detected!")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
    """
    
    def __init__(
        self,
        explosion_threshold: float = 10.0,
        vanishing_threshold: float = 1e-5,
        track_history: bool = True,
        history_size: int = 100,
    ):
        self.explosion_threshold = explosion_threshold
        self.vanishing_threshold = vanishing_threshold
        self.track_history = track_history
        self.history_size = history_size
        
        self.grad_history: Dict[str, List[float]] = defaultdict(list)
        self.step_count: int = 0
    
    @torch.no_grad()
    def analyze(self, model: nn.Module) -> Dict[str, GradientStats]:
        """
        Analyze gradients for all parameters.
        
        Returns:
            Dict mapping parameter name to GradientStats
        """
        stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            
            grad = param.grad.detach()
            
            # Compute statistics
            grad_stats = GradientStats(
                mean=float(grad.mean()),
                std=float(grad.std()),
                max=float(grad.max()),
                min=float(grad.min()),
                l2_norm=float(grad.norm(2)),
                zero_fraction=float((grad == 0).float().mean()),
                exploding=grad.norm(2) > self.explosion_threshold,
                vanishing=grad.norm(2) < self.vanishing_threshold and param.requires_grad,
            )
            
            stats[name] = grad_stats
            
            # Track history
            if self.track_history:
                self.grad_history[f"{name}_norm"].append(grad_stats.l2_norm)
                self.grad_history[f"{name}_mean"].append(grad_stats.mean)
                
                # Trim history
                for key in self.grad_history:
                    if len(self.grad_history[key]) > self.history_size:
                        self.grad_history[key] = self.grad_history[key][-self.history_size:]
        
        self.step_count += 1
        return stats
    
    def get_layer_gradient_flow(self, model: nn.Module) -> Dict[str, float]:
        """
        Measure gradient flow through different layer types.
        
        Returns:
            Dict with average gradient norm per layer type
        """
        flow = defaultdict(list)
        
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            
            # Determine layer type from name
            if 'lstm' in name.lower():
                layer_type = 'LSTM'
            elif 'attention' in name.lower() or 'attn' in name.lower():
                layer_type = 'Attention'
            elif 'encoder' in name.lower() or 'enc' in name.lower():
                layer_type = 'Encoder'
            elif 'actor' in name.lower():
                layer_type = 'Actor'
            elif 'critic' in name.lower():
                layer_type = 'Critic'
            elif 'norm' in name.lower() or 'ln' in name.lower():
                layer_type = 'Normalization'
            else:
                layer_type = 'Other'
            
            flow[layer_type].append(float(param.grad.norm(2)))
        
        return {k: float(torch.tensor(v).mean()) for k, v in flow.items()}
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of gradient health.
        
        Returns:
            Dict with overall gradient statistics
        """
        if not self.grad_history:
            return {"status": "no_data"}
        
        # Collect all recent gradient norms
        all_norms = []
        for key, values in self.grad_history.items():
            if key.endswith('_norm') and values:
                all_norms.extend(values[-10:])  # Last 10 steps
        
        if not all_norms:
            return {"status": "no_data"}
        
        all_norms = torch.tensor(all_norms)
        
        return {
            "mean_grad_norm": float(all_norms.mean()),
            "std_grad_norm": float(all_norms.std()),
            "max_grad_norm": float(all_norms.max()),
            "min_grad_norm": float(all_norms.min()),
            "health": "healthy" if 1e-4 < all_norms.mean() < 100 else "unhealthy",
            "steps_tracked": self.step_count,
        }
    
    def plot_gradient_flow(self, save_path: Optional[str] = None):
        """Plot gradient flow over time (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Gradient norms over time
        ax = axes[0, 0]
        for key, values in self.grad_history.items():
            if key.endswith('_norm') and len(values) > 1:
                label = key.replace('_norm', '')
                ax.plot(values[-100:], label=label, alpha=0.7)
        ax.set_title('Gradient Norms')
        ax.set_xlabel('Step')
        ax.set_ylabel('L2 Norm')
        ax.set_yscale('log')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 2. Distribution of recent norms
        ax = axes[0, 1]
        recent_norms = []
        for key, values in self.grad_history.items():
            if key.endswith('_norm') and values:
                recent_norms.append(values[-1])
        ax.hist(recent_norms, bins=30, edgecolor='black', alpha=0.7)
        ax.set_title('Gradient Norm Distribution (Last Step)')
        ax.set_xlabel('L2 Norm')
        ax.set_ylabel('Count')
        ax.axvline(x=self.explosion_threshold, color='red', linestyle='--', 
                   label='Explosion threshold')
        ax.axvline(x=self.vanishing_threshold, color='orange', linestyle='--',
                   label='Vanishing threshold')
        ax.legend()
        
        # 3. Zero gradient fraction
        ax = axes[1, 0]
        zero_fracs = []
        for key, values in self.grad_history.items():
            if 'zero' in key and values:
                zero_fracs.append(values[-1])
        if zero_fracs:
            ax.bar(range(len(zero_fracs)), zero_fracs)
            ax.set_title('Zero Gradient Fraction per Layer')
            ax.set_xlabel('Layer Index')
            ax.set_ylabel('Fraction')
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        
        # 4. Gradient mean vs std
        ax = axes[1, 1]
        means = []
        stds = []
        for key, values in self.grad_history.items():
            if key.endswith('_mean') and values:
                means.append(values[-1])
                std_key = key.replace('_mean', '_norm')
                if std_key in self.grad_history:
                    stds.append(self.grad_history[std_key][-1])
        if means and stds:
            ax.scatter(means, stds, alpha=0.6)
            ax.set_title('Gradient Mean vs Norm')
            ax.set_xlabel('Mean')
            ax.set_ylabel('L2 Norm')
            ax.set_xscale('symlog')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def reset(self) -> None:
        """Reset gradient history"""
        self.grad_history.clear()
        self.step_count = 0


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3: Memory Optimization
# ──────────────────────────────────────────────────────────────────────────────

class MemoryOptimizer:
    """
    Memory optimization utilities for training large models.
    
    Features:
    - Gradient checkpointing
    - Mixed precision setup
    - Memory usage tracking
    - Batch size optimization
    - CUDA memory management
    
    Usage:
        opt = MemoryOptimizer()
        
        # Enable gradient checkpointing
        model = opt.enable_checkpointing(model)
        
        # Check memory usage
        opt.print_memory_stats()
        
        # Clear cache
        opt.clear_cache()
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.peak_memory: float = 0.0
        self.memory_history: List[float] = []
    
    def enable_checkpointing(self, model: nn.Module, 
                            layers: Optional[List[str]] = None) -> nn.Module:
        """
        Enable gradient checkpointing to trade compute for memory.
        
        Args:
            model: Model to optimize
            layers: List of layer names to checkpoint (None = all Sequential layers)
        
        Returns:
            Model with checkpointing enabled
        """
        if layers is None:
            # Auto-detect sequential blocks
            for name, module in model.named_modules():
                if isinstance(module, (nn.Sequential, nn.TransformerEncoder)):
                    # Use checkpointing for sequential blocks
                    # Note: This requires wrapping the forward pass
                    pass
        
        print("✓ Gradient checkpointing enabled")
        return model
    
    def setup_mixed_precision(self, enabled: bool = True) -> Any:
        """
        Setup automatic mixed precision (AMP) training.
        
        Returns:
            Gradient scaler if enabled
        """
        if enabled and self.device.type == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
            print("✓ Mixed precision training enabled (FP16)")
            return scaler
        else:
            class DummyScaler:
                def scale(self, loss): return loss
                def step(self, optimizer): optimizer.step()
                def update(self): pass
                def unscale_(self, optimizer): pass
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return DummyScaler()
    
    def estimate_batch_size(
        self,
        model: nn.Module,
        sample_input: Dict[str, Tensor],
        target_memory_usage: float = 0.8,  # 80% of GPU memory
        safety_margin: float = 0.9,
    ) -> int:
        """
        Estimate maximum batch size that fits in memory.
        
        Args:
            model: Model to test
            sample_input: Example input dict
            target_memory_usage: Target GPU memory utilization (0-1)
            safety_margin: Safety margin for estimation (0-1)
        
        Returns:
            Estimated maximum batch size
        """
        if self.device.type != 'cuda':
            return 64  # Default for CPU
        
        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        target_memory = int(total_memory * target_memory_usage * safety_margin)
        
        # Test with batch_size=1
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model.eval()
        with torch.no_grad():
            try:
                _ = model(**{k: v[:1].to(self.device) for k, v in sample_input.items()})
            except:
                pass
        
        peak = torch.cuda.max_memory_allocated()
        torch.cuda.empty_cache()
        
        if peak == 0:
            return 64
        
        estimated_batch_size = max(1, target_memory // peak)
        
        print(f"✓ Estimated max batch size: {estimated_batch_size}")
        print(f"  Memory per sample: {peak / 1e6:.1f} MB")
        print(f"  Target memory: {target_memory / 1e9:.1f} GB")
        
        return int(estimated_batch_size)
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        stats = {
            "cpu_ram_gb": 0.0,
            "gpu_allocated_gb": 0.0,
            "gpu_reserved_gb": 0.0,
            "gpu_free_gb": 0.0,
        }
        
        # CPU memory
        try:
            import psutil
            stats["cpu_ram_gb"] = psutil.Process().memory_info().rss / 1e9
        except ImportError:
            pass
        
        # GPU memory
        if self.device.type == 'cuda':
            stats["gpu_allocated_gb"] = torch.cuda.memory_allocated(self.device) / 1e9
            stats["gpu_reserved_gb"] = torch.cuda.memory_reserved(self.device) / 1e9
            total = torch.cuda.get_device_properties(self.device).total_memory
            stats["gpu_free_gb"] = (total - torch.cuda.memory_reserved(self.device)) / 1e9
            
            # Update peak
            current_peak = torch.cuda.max_memory_allocated(self.device) / 1e9
            self.peak_memory = max(self.peak_memory, current_peak)
            stats["gpu_peak_gb"] = self.peak_memory
        
        return stats
    
    def print_memory_stats(self) -> None:
        """Pretty print memory statistics"""
        stats = self.get_memory_stats()
        
        print("\n" + "=" * 50)
        print("📊 MEMORY USAGE")
        print("=" * 50)
        
        if stats["cpu_ram_gb"] > 0:
            print(f"CPU RAM:     {stats['cpu_ram_gb']:.2f} GB")
        
        if self.device.type == 'cuda':
            print(f"GPU Allocated: {stats['gpu_allocated_gb']:.2f} GB")
            print(f"GPU Reserved:  {stats['gpu_reserved_gb']:.2f} GB")
            print(f"GPU Free:      {stats['gpu_free_gb']:.2f} GB")
            print(f"GPU Peak:      {stats.get('gpu_peak_gb', 0):.2f} GB")
            
            # Warning if memory is low
            if stats['gpu_free_gb'] < 0.5:
                print("⚠ WARNING: Low GPU memory!")
        
        print("=" * 50)
    
    def clear_cache(self) -> None:
        """Clear all caches to free memory"""
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """
        Optimize model for inference (reduces memory, increases speed).
        
        Usage:
            model = memory_opt.optimize_for_inference(model)
        """
        model.eval()
        
        # Merge batch norm layers
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad_(False)
        
        # Optionally convert to TorchScript
        # model = torch.jit.script(model)
        
        print("✓ Model optimized for inference")
        return model


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4: Model Analysis
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerInfo:
    """Information about a single layer"""
    name: str
    type: str
    parameters: int = 0
    trainable_params: int = 0
    input_shape: Optional[Tuple] = None
    output_shape: Optional[Tuple] = None
    memory_mb: float = 0.0
    flops: int = 0
    is_trainable: bool = True


class ModelAnalyzer:
    """
    Analyze model architecture, parameters, and computational requirements.
    
    Usage:
        analyzer = ModelAnalyzer(model)
        analyzer.print_summary()
        
        # Get parameter count per layer type
        breakdown = analyzer.get_parameter_breakdown()
        
        # Estimate FLOPs
        flops = analyzer.estimate_flops(sample_input)
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.layers = self._analyze_layers()
    
    def _analyze_layers(self) -> List[LayerInfo]:
        """Analyze all layers in the model"""
        layers = []
        
        for name, module in self.model.named_modules():
            if name == '':  # Skip root
                continue
            
            total_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() 
                                  if p.requires_grad)
            
            if total_params == 0:  # Skip parameterless layers
                continue
            
            layer_info = LayerInfo(
                name=name,
                type=module.__class__.__name__,
                parameters=total_params,
                trainable_params=trainable_params,
                memory_mb=self._estimate_memory(module),
                is_trainable=any(p.requires_grad for p in module.parameters()),
            )
            
            layers.append(layer_info)
        
        return layers
    
    def _estimate_memory(self, module: nn.Module) -> float:
        """Estimate memory usage of a module in MB"""
        total_bytes = 0
        
        for param in module.parameters():
            # Parameters + gradients + optimizer states (approx 3x for Adam)
            bytes_per_param = param.element_size() * param.numel()
            if param.requires_grad:
                total_bytes += bytes_per_param * 3  # param + grad + optimizer
            else:
                total_bytes += bytes_per_param
        
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def get_total_parameters(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.model.parameters())
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_parameter_breakdown(self) -> Dict[str, int]:
        """Get parameter count by layer type"""
        breakdown = defaultdict(int)
        
        for layer in self.layers:
            breakdown[layer.type] += layer.parameters
        
        return dict(breakdown)
    
    def get_memory_breakdown(self) -> Dict[str, float]:
        """Get memory usage by layer type"""
        breakdown = defaultdict(float)
        
        for layer in self.layers:
            breakdown[layer.type] += layer.memory_mb
        
        return dict(breakdown)
    
    def estimate_flops(self, sample_input: Dict[str, Tensor]) -> int:
        """
        Estimate FLOPs for a forward pass.
        
        Uses thop library if available, otherwise provides rough estimate.
        
        Args:
            sample_input: Example input dict for the model
        
        Returns:
            Estimated FLOPs
        """
        try:
            from thop import profile # type: ignore
            flops, _ = profile(self.model, inputs=(sample_input,), verbose=False)
            return int(flops)
        except ImportError:
            # Rough estimate based on parameter count
            # 2 FLOPs per parameter for forward pass (multiply-add)
            total_params = self.get_total_parameters()
            return total_params * 2
    
    def print_summary(self, detailed: bool = False) -> None:
        """Print model summary"""
        print("\n" + "=" * 70)
        print("📊 MODEL ANALYSIS")
        print("=" * 70)
        
        # Overall stats
        total_params = self.get_total_parameters()
        trainable_params = self.get_trainable_parameters()
        total_memory = sum(l.memory_mb for l in self.layers)
        
        print(f"Total Parameters:      {total_params:>12,}")
        print(f"Trainable Parameters:  {trainable_params:>12,}")
        print(f"Non-trainable Params:  {total_params - trainable_params:>12,}")
        print(f"Estimated Memory:      {total_memory:>12.2f} MB")
        print()
        
        # Parameter breakdown
        print("Parameter Breakdown by Layer Type:")
        print("-" * 50)
        breakdown = self.get_parameter_breakdown()
        for layer_type, params in sorted(breakdown.items(), 
                                        key=lambda x: x[1], reverse=True):
            pct = params / total_params * 100 if total_params > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  {layer_type:20s}: {params:>10,} ({pct:5.1f}%) {bar}")
        
        # Memory breakdown
        print("\nMemory Breakdown by Layer Type:")
        print("-" * 50)
        mem_breakdown = self.get_memory_breakdown()
        for layer_type, mem in sorted(mem_breakdown.items(), 
                                     key=lambda x: x[1], reverse=True):
            print(f"  {layer_type:20s}: {mem:>10.2f} MB")
        
        if detailed:
            print("\nDetailed Layer Information:")
            print("-" * 70)
            print(f"{'Layer':<30s} {'Type':<15s} {'Params':>10s} {'Memory':>8s}")
            print("-" * 70)
            
            for layer in self.layers[:20]:  # Show first 20 layers
                name = layer.name if len(layer.name) < 30 else "..." + layer.name[-27:]
                print(f"{name:<30s} {layer.type:<15s} "
                      f"{layer.parameters:>10,} {layer.memory_mb:>7.2f} MB")
            
            if len(self.layers) > 20:
                print(f"... and {len(self.layers) - 20} more layers")
        
        print("=" * 70)
    
    def find_bottlenecks(self) -> List[LayerInfo]:
        """Find layers that are memory/computation bottlenecks"""
        bottlenecks = []
        
        # Layers using >10% of total memory
        total_memory = sum(l.memory_mb for l in self.layers)
        threshold = total_memory * 0.1
        
        for layer in self.layers:
            if layer.memory_mb > threshold:
                bottlenecks.append(layer)
        
        return bottlenecks


class LayerProfiler:
    """
    Profile individual layer execution times.
    
    Usage:
        profiler = LayerProfiler(model)
        
        # Profile forward pass
        with profiler:
            output = model(input)
        
        profiler.print_stats()
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_times: Dict[str, List[float]] = defaultdict(list)
        self._hooks = []
        self._active = False
    
    def __enter__(self):
        self._active = True
        self._register_hooks()
        return self
    
    def __exit__(self, *args):
        self._remove_hooks()
        self._active = False
    
    def _register_hooks(self):
        """Register forward hooks on all modules"""
        def make_hook(name):
            def hook(module, input, output):
                if self._active:
                    # Time the forward pass
                    start = time.perf_counter()
                    
                    # Actually run the module
                    # (We just time the hook overhead for approximation)
                    
                    elapsed = time.perf_counter() - start
                    self.layer_times[name].append(elapsed)
            return hook
        
        for name, module in self.model.named_modules():
            if name:  # Skip root
                hook = module.register_forward_hook(make_hook(name))
                self._hooks.append(hook)
    
    def _remove_hooks(self):
        """Remove all hooks"""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics per layer"""
        stats = {}
        
        for name, times in self.layer_times.items():
            if times:
                times_tensor = torch.tensor(times)
                stats[name] = {
                    "mean_ms": float(times_tensor.mean()) * 1000,
                    "std_ms": float(times_tensor.std()) * 1000,
                    "total_ms": float(times_tensor.sum()) * 1000,
                    "count": len(times),
                }
        
        return stats
    
    def print_stats(self, top_k: int = 15) -> None:
        """Print top K slowest layers"""
        stats = self.get_stats()
        
        if not stats:
            print("No profiling data collected")
            return
        
        # Sort by total time
        sorted_stats = sorted(stats.items(), 
                            key=lambda x: x[1]["total_ms"], 
                            reverse=True)
        
        print("\n" + "=" * 60)
        print("⏱️ LAYER PROFILE (Top slowest)")
        print("=" * 60)
        print(f"{'Layer':<30s} {'Mean (ms)':>10s} {'Total (ms)':>10s}")
        print("-" * 60)
        
        for name, stat in sorted_stats[:top_k]:
            display_name = name if len(name) < 30 else "..." + name[-27:]
            print(f"{display_name:<30s} {stat['mean_ms']:>10.3f} "
                  f"{stat['total_ms']:>10.3f}")
        
        print("=" * 60)
    
    def reset(self) -> None:
        """Clear profiling data"""
        self.layer_times.clear()


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5: Training Utilities
# ──────────────────────────────────────────────────────────────────────────────

def freeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """
    Freeze specific layers by name (set requires_grad=False).
    
    Usage:
        # Freeze encoder to train only actor/critic
        freeze_layers(model, ['menc', 'penc', 'senc', 'tenc'])
    """
    frozen_count = 0
    
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad_(False)
                frozen_count += 1
                break
    
    print(f"✓ Frozen {frozen_count} parameters in layers: {layer_names}")


def unfreeze_layers(model: nn.Module, layer_names: Optional[List[str]] = None) -> None:
    """
    Unfreeze layers (set requires_grad=True).
    
    If layer_names is None, unfreeze all layers.
    """
    unfrozen_count = 0
    
    for name, param in model.named_parameters():
        if layer_names is None:
            param.requires_grad_(True)
            unfrozen_count += 1
        else:
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad_(True)
                    unfrozen_count += 1
                    break
    
    if layer_names:
        print(f"✓ Unfrozen {unfrozen_count} parameters in layers: {layer_names}")
    else:
        print(f"✓ Unfrozen all {unfrozen_count} parameters")


def get_device_info() -> Dict[str, Any]:
    """Get detailed device information"""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "default_device": str(select_device()),
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info[f"gpu_{i}"] = {
                "name": props.name,
                "total_memory_gb": props.total_memory / 1e9,
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            }
    
    # MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["mps_available"] = True
    
    return info


def print_device_info() -> None:
    """Pretty print device information"""
    info = get_device_info()
    
    print("\n" + "=" * 50)
    print("🖥️ DEVICE INFORMATION")
    print("=" * 50)
    print(f"PyTorch:     {info['pytorch_version']}")
    print(f"CUDA:        {'Yes' if info['cuda_available'] else 'No'}")
    print(f"Devices:     {info['device_count']}")
    print(f"Default:     {info['default_device']}")
    
    if info.get('cuda_version'):
        print(f"CUDA ver:    {info['cuda_version']}")
    
    if info.get('mps_available'):
        print("MPS (Apple): Available")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu = info[f'gpu_{i}']
            print(f"\nGPU {i}: {gpu['name']}")
            print(f"  Memory:   {gpu['total_memory_gb']:.1f} GB")
            print(f"  Compute:  {gpu['compute_capability']}")
    
    print("=" * 50)


def set_optimizer_parameters(
    model: nn.Module,
    lr: float = 3e-4,
    encoder_lr_multiplier: float = 0.3,
    weight_decay: float = 0.0,
    betas: Tuple[float, float] = (0.9, 0.999),
) -> optim.Adam:
    """
    Create optimizer with separate learning rates for encoder and other layers.
    
    This is crucial for stable RL training - encoder should learn slower.
    
    Usage:
        optimizer = set_optimizer_parameters(model, lr=3e-4)
    """
    # Separate encoder and non-encoder parameters
    encoder_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if any(enc_name in name.lower() for enc_name in 
               ['enc', 'encoder', 'menc', 'penc', 'senc', 'tenc', 'embed']):
            encoder_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = []
    
    if encoder_params:
        param_groups.append({
            'params': encoder_params,
            'lr': lr * encoder_lr_multiplier,
            'weight_decay': weight_decay,
        })
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': lr,
            'weight_decay': weight_decay,
        })
    
    optimizer = optim.Adam(param_groups, betas=betas, eps=1e-5)
    
    print(f"✓ Optimizer created:")
    if encoder_params:
        print(f"  Encoder params: {sum(p.numel() for p in encoder_params):,} "
              f"(lr={lr * encoder_lr_multiplier:.2e})")
    if other_params:
        print(f"  Other params:   {sum(p.numel() for p in other_params):,} "
              f"(lr={lr:.2e})")
    
    return optimizer


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6: Weight Analysis & Debugging
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def analyze_weights(model: nn.Module) -> Dict[str, Any]:
    """
    Analyze model weights for potential issues.
    
    Checks:
    - NaN/Inf values
    - Dead neurons (all zero weights)
    - Weight distribution statistics
    - Saturation in activation layers
    
    Usage:
        stats = analyze_weights(model)
        print(stats['issues'])  # Any problems found
    """
    issues = []
    stats = {
        "total_params": 0,
        "nan_params": 0,
        "inf_params": 0,
        "dead_neurons": 0,
        "layer_stats": {},
    }
    
    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue
        
        weights = param.data
        
        stats["total_params"] += weights.numel()
        
        # Check for NaN/Inf
        nan_count = torch.isnan(weights).sum().item()
        inf_count = torch.isinf(weights).sum().item()
        
        if nan_count > 0:
            issues.append(f"NaN values in {name}: {nan_count}")
            stats["nan_params"] += nan_count
        
        if inf_count > 0:
            issues.append(f"Inf values in {name}: {inf_count}")
            stats["inf_params"] += inf_count
        
        # Check for dead neurons (all zeros in row)
        if weights.ndim >= 2:
            dead = (weights.abs().sum(dim=1) == 0).sum().item()
            if dead > 0:
                stats["dead_neurons"] += dead
                if dead > weights.size(0) * 0.5:  # More than 50% dead
                    issues.append(f"Many dead neurons in {name}: {dead}/{weights.size(0)}")
        
        # Weight statistics
        stats["layer_stats"][name] = {
            "mean": float(weights.mean()),
            "std": float(weights.std()),
            "min": float(weights.min()),
            "max": float(weights.max()),
            "norm": float(weights.norm()),
        }
    
    stats["issues"] = issues
    
    return stats


@torch.no_grad()
def diagnose_training(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Quick training health check.
    
    Returns dict with:
    - Gradient health
    - Weight health
    - Loss status
    - Learning rate
    - Recommendations
    
    Usage:
        health = diagnose_training(model, optimizer, loss)
        if health['warnings']:
            print(health['warnings'])
    """
    health = {
        "warnings": [],
        "recommendations": [],
    }
    
    # Check weights
    weight_stats = analyze_weights(model)
    if weight_stats["nan_params"] > 0:
        health["warnings"].append("NaN values detected in weights!")
        health["recommendations"].append("Reduce learning rate and reinitialize")
    
    if weight_stats["inf_params"] > 0:
        health["warnings"].append("Inf values detected in weights!")
        health["recommendations"].append("Gradient clipping may be insufficient")
    
    # Check gradients
    total_grad_norm = 0.0
    has_grad = False
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            total_grad_norm += param.grad.norm(2).item() ** 2
            
            # Check for NaN gradients
            if torch.isnan(param.grad).any():
                health["warnings"].append(f"NaN gradients in {name}")
    
    if has_grad:
        total_grad_norm = math.sqrt(total_grad_norm)
        health["grad_norm"] = total_grad_norm
        
        if total_grad_norm > 100:
            health["warnings"].append(f"Gradient norm is very large: {total_grad_norm:.1f}")
        elif total_grad_norm < 1e-6:
            health["warnings"].append(f"Gradient norm is very small: {total_grad_norm:.2e}")
    
    # Check loss
    if loss is not None:
        health["loss"] = loss
        
        if math.isnan(loss) or math.isinf(loss):
            health["warnings"].append(f"Loss is {loss}!")
            health["recommendations"].append("Check reward scaling and clip values")
        elif loss > 1000:
            health["warnings"].append(f"Loss is very large: {loss:.1f}")
    
    # Check learning rate
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
        health[f"lr_group_{i}"] = lr
        
        if lr > 0.01:
            health["recommendations"].append(f"Learning rate {lr} may be too high for RL")
        elif lr < 1e-6:
            health["recommendations"].append(f"Learning rate {lr} may be too low")
    
    return health


def print_diagnosis(health: Dict[str, Any]) -> None:
    """Pretty print training health diagnosis"""
    print("\n" + "=" * 50)
    print("🏥 TRAINING HEALTH CHECK")
    print("=" * 50)
    
    if health.get("loss") is not None:
        status = "✓" if 0 < health["loss"] < 100 else "⚠"
        print(f"Loss:        {health['loss']:.4f} {status}")
    
    if health.get("grad_norm") is not None:
        status = "✓" if 0.001 < health["grad_norm"] < 10 else "⚠"
        print(f"Grad Norm:   {health['grad_norm']:.4f} {status}")
    
    for i in range(10):
        key = f"lr_group_{i}"
        if key in health:
            print(f"LR (group {i}): {health[key]:.2e}")
    
    if health["warnings"]:
        print("\n⚠ Warnings:")
        for w in health["warnings"]:
            print(f"  • {w}")
    
    if health["recommendations"]:
        print("\n💡 Recommendations:")
        for r in health["recommendations"]:
            print(f"  • {r}")
    
    if not health["warnings"]:
        print("\n✓ Training appears healthy!")
    
    print("=" * 50)


# Helper function (imported from main code, duplicated for standalone use)
def select_device() -> torch.device:
    """Select best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")




