
from __future__ import annotations

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Try importing plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Visualization disabled.")


def plot_training_curves(metrics_history: List[Dict],
                         figsize: Tuple[int, int] = (15, 10),
                         save_path: Optional[str] = None):
    """
    Plot comprehensive training curves.
    
    Args:
        metrics_history: List of metric dicts from training epochs
        figsize: Figure size
        save_path: If provided, save figure to this path
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return

    if not metrics_history:
        print("No metrics to plot")
        return

    # Collect metrics
    epochs = range(1, len(metrics_history) + 1)

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle('Training Diagnostics', fontsize=14, fontweight='bold')

    # 1. Losses
    ax = axes[0, 0]
    if 'policy_loss' in metrics_history[0]:
        ax.plot(epochs, [m['policy_loss'] for m in metrics_history],
                label='Policy Loss', alpha=0.8)
    if 'value_loss' in metrics_history[0]:
        ax.plot(epochs, [m['value_loss'] for m in metrics_history],
                label='Value Loss', alpha=0.8)
    if 'total_loss' in metrics_history[0]:
        ax.plot(epochs, [m['total_loss'] for m in metrics_history],
                label='Total Loss', alpha=0.8, linewidth=2, color='black')
    ax.set_title('Losses')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Rewards
    ax = axes[0, 1]
    if 'mean_reward' in metrics_history[0]:
        rewards = [m['mean_reward'] for m in metrics_history]
        ax.plot(epochs, rewards, label='Mean Reward',
                color='green', linewidth=2)
        # Add rolling average
        if len(rewards) > 10:
            window = min(10, len(rewards) // 4)
            rolling = np.convolve(rewards, np.ones(
                window)/window, mode='valid')
            ax.plot(epochs[window-1:], rolling, '--',
                    label=f'{window}-epoch MA', color='darkgreen', alpha=0.7)
    ax.set_title('Episode Rewards')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. KL Divergence
    ax = axes[1, 0]
    if 'approx_kl' in metrics_history[0]:
        kl_values = [m['approx_kl'] for m in metrics_history]
        ax.plot(epochs, kl_values, label='KL Divergence', color='purple')
        if 'target_kl' in metrics_history[0]:
            ax.axhline(y=metrics_history[0].get('target_kl', 0.02),
                       color='red', linestyle='--', label='Target KL')
        ax.set_title('Policy KL Divergence')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 4. Entropy
    ax = axes[1, 1]
    if 'entropy' in metrics_history[0]:
        ax.plot(epochs, [m['entropy'] for m in metrics_history],
                label='Policy Entropy', color='orange')
    ax.set_title('Policy Entropy')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Clip Fraction
    ax = axes[2, 0]
    if 'clip_frac' in metrics_history[0]:
        ax.plot(epochs, [m['clip_frac'] for m in metrics_history],
                label='Clip Fraction', color='red')
        ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('PPO Clip Fraction')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Gradient Norm
    ax = axes[2, 1]
    if 'grad_norm' in metrics_history[0]:
        ax.plot(epochs, [m['grad_norm'] for m in metrics_history],
                label='Gradient Norm', color='brown')
        ax.set_yscale('log')
    ax.set_title('Gradient Norm (log scale)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_reward_decomposition(reward_components: Dict[str, List[float]],
                              figsize: Tuple[int, int] = (12, 8),
                              save_path: Optional[str] = None):
    """
    Plot breakdown of reward components over time.
    
    Args:
        reward_components: Dict mapping component name to list of values
        figsize: Figure size
        save_path: If provided, save figure to this path
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # 1. Stacked area plot
    ax = axes[0]
    components = list(reward_components.keys())

    # Calculate steps
    n_steps = min(len(v) for v in reward_components.values())
    steps = np.arange(n_steps)

    # Create stacked data
    data = np.zeros((len(components), n_steps))
    for i, comp in enumerate(components):
        data[i] = reward_components[comp][:n_steps]

    # Separate positive and negative
    pos_data = np.maximum(data, 0)
    neg_data = np.minimum(data, 0)

    # Plot positive components
    ax.stackplot(steps, pos_data, labels=components, alpha=0.7)

    # Plot negative components
    ax.stackplot(steps, neg_data, alpha=0.7, colors=['red']*len(components))

    ax.set_title('Reward Component Decomposition')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)

    # 2. Dominance over time (rolling)
    ax = axes[1]
    window = max(n_steps // 20, 50)

    for comp in components:
        values = np.array(reward_components[comp][:n_steps])
        abs_values = np.abs(values)
        # Rolling mean of absolute values
        if len(values) >= window:
            rolling = np.convolve(abs_values, np.ones(
                window)/window, mode='valid')
            roll_steps = steps[window-1:]
            # Normalize by total
            total_rolling = sum(
                np.convolve(np.abs(np.array(reward_components[c][:n_steps])),
                            np.ones(window)/window, mode='valid')
                for c in components
            )
            dominance = rolling / (total_rolling + 1e-8)
            ax.plot(roll_steps, dominance, label=comp, alpha=0.8)

    ax.set_title('Reward Component Dominance (Rolling)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Fraction of Total |Reward|')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


class Dashboard:
    """
    Simple training dashboard for monitoring progress.
    
    Usage:
        dashboard = Dashboard()
        
        for epoch in range(n_epochs):
            metrics = train_epoch()
            dashboard.update(metrics)
            
            if epoch % 10 == 0:
                dashboard.display()
    """

    def __init__(self):
        self.metrics_history: List[Dict] = []
        self._reward_window: List[float] = []
        self._best_metric: float = float('-inf')
        self._best_epoch: int = 0

    def update(self, metrics: Dict) -> None:
        """Update dashboard with new metrics"""
        self.metrics_history.append(metrics)

        if 'mean_reward' in metrics:
            self._reward_window.append(metrics['mean_reward'])
            if len(self._reward_window) > 100:
                self._reward_window.pop(0)

    def display(self) -> None:
        """Print current training status"""
        if not self.metrics_history:
            print("No metrics to display")
            return

        latest = self.metrics_history[-1]
        epoch = len(self.metrics_history)

        print("\n" + "=" * 60)
        print(f"📊 TRAINING DASHBOARD — Epoch {epoch}")
        print("=" * 60)

        # Losses
        if 'total_loss' in latest:
            print(f"📉 Total Loss:     {latest['total_loss']:.4f}")
        if 'policy_loss' in latest:
            print(f"   Policy Loss:   {latest['policy_loss']:.4f}")
        if 'value_loss' in latest:
            print(f"   Value Loss:    {latest['value_loss']:.4f}")

        # Rewards
        if 'mean_reward' in latest:
            avg_reward = np.mean(
                self._reward_window[-10:]) if self._reward_window else 0
            print(f"\n💰 Mean Reward:    {latest['mean_reward']:+.4f}")
            print(f"   10-epoch avg:  {avg_reward:+.4f}")

        # Policy metrics
        if 'approx_kl' in latest:
            kl = latest['approx_kl']
            status = "✓" if kl < 0.02 else "⚠" if kl < 0.05 else "✗"
            print(f"\n📐 KL Divergence:  {kl:.5f} {status}")

        if 'entropy' in latest:
            ent = latest['entropy']
            status = "✓" if ent > 0.01 else "⚠"
            print(f"🎲 Entropy:        {ent:.4f} {status}")

        if 'clip_frac' in latest:
            cf = latest['clip_frac']
            status = "✓" if cf < 0.2 else "⚠"
            print(f"📎 Clip Fraction:  {cf:.3f} {status}")

        # Gradient
        if 'grad_norm' in latest:
            gn = latest['grad_norm']
            status = "✓" if gn < 10 else "⚠" if gn < 50 else "✗"
            print(f"\n📏 Grad Norm:      {gn:.3f} {status}")

        # Learning rate
        if 'lr' in latest:
            print(f"🔧 Learning Rate:  {latest['lr']:.2e}")

        print("=" * 60)

    def get_summary(self) -> str:
        """Get text summary of training"""
        if not self.metrics_history:
            return "No training data"

        recent = self.metrics_history[-10:]

        lines = [
            "TRAINING SUMMARY",
            f"Total epochs: {len(self.metrics_history)}",
            f"Avg loss (last 10): {np.mean([m.get('total_loss', 0) for m in recent]):.4f}",
            f"Avg reward (last 10): {np.mean([m.get('mean_reward', 0) for m in recent]):+.4f}",
        ]

        return "\n".join(lines)
