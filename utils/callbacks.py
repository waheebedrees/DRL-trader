"""
Training Callbacks
==================
Callback system for training loop management.

Features:
- Early stopping based on various criteria
- Model checkpointing (best, periodic)
- Progress logging
- WandB/MLflow integration hooks
"""

from __future__ import annotations

import os
import json
import torch
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime


class EarlyStopping:
    """
    Early stopping with patience and multiple criteria.
    
    Usage:
        stopper = EarlyStopping(patience=20, min_delta=0.001, mode='max')
        
        for epoch in range(n_epochs):
            metrics = train_epoch()
            
            should_stop, reason = stopper.check(metrics['mean_reward'])
            if should_stop:
                print(f"Early stopping: {reason}")
                break
    """

    def __init__(self, patience: int = 20, min_delta: float = 0.001,
                 mode: str = 'max', restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # 'max' for rewards, 'min' for losses
        self.restore_best = restore_best

        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.counter: int = 0
        self.should_stop: bool = False
        self.best_state: Optional[Dict] = None

    def check(self, metric: float, model: Optional[torch.nn.Module] = None) -> Tuple[bool, str]:
        """
        Check if training should stop.
        
        Args:
            metric: Current metric value
            model: If provided and restore_best=True, saves best model state
        
        Returns:
            (should_stop, reason) tuple
        """
        if self.best_score is None:
            self.best_score = metric
            self.best_epoch = 0
            if model and self.restore_best:
                self.best_state = {k: v.cpu().clone()
                                   for k, v in model.state_dict().items()}
            return False, "first_epoch"

        # Check improvement
        if self.mode == 'max':
            improved = metric > self.best_score + self.min_delta
        else:
            improved = metric < self.best_score - self.min_delta

        if improved:
            self.best_score = metric
            self.counter = 0
            if model and self.restore_best:
                self.best_state = {k: v.cpu().clone()
                                   for k, v in model.state_dict().items()}
            return False, "improved"
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True, f"no_improvement_for_{self.patience}_epochs"
            return False, f"no_improvement_{self.counter}/{self.patience}"

    def restore(self, model: torch.nn.Module) -> None:
        """Restore model to best state"""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

    def reset(self) -> None:
        """Reset early stopping state"""
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False
        self.best_state = None


class ModelCheckpoint:
    """
    Save model checkpoints during training.
    
    Usage:
        checkpoint = ModelCheckpoint(save_dir='checkpoints', save_best_only=True)
        
        for epoch in range(n_epochs):
            train_epoch()
            checkpoint.save(model, epoch, metrics)
    """

    def __init__(self, save_dir: str = 'checkpoints',
                 save_best_only: bool = True,
                 save_freq: int = 50,
                 max_checkpoints: int = 5,
                 metric_name: str = 'mean_reward',
                 mode: str = 'max'):
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.mode = mode

        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        self.saved_checkpoints: List[str] = []

        os.makedirs(save_dir, exist_ok=True)

    def save(self, model: torch.nn.Module, epoch: int,
             metrics: Dict[str, float],
             trainer: Optional[Any] = None) -> Optional[str]:
        """
        Save checkpoint if conditions are met.
        
        Returns:
            Path to saved checkpoint, or None if not saved
        """
        should_save = False

        if self.save_best_only:
            current_metric = metrics.get(self.metric_name, 0)

            if self.mode == 'max' and current_metric > self.best_metric:
                self.best_metric = current_metric
                should_save = True
            elif self.mode == 'min' and current_metric < self.best_metric:
                self.best_metric = current_metric
                should_save = True
        else:
            if epoch % self.save_freq == 0:
                should_save = True

        if not should_save:
            return None

        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }

        if trainer is not None:
            if hasattr(trainer, 'opt'):
                checkpoint['optimizer_state_dict'] = trainer.opt.state_dict()

        # Save
        filename = f"checkpoint_epoch_{epoch:04d}.pt"
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)

        self.saved_checkpoints.append(filepath)

        # Remove old checkpoints
        while len(self.saved_checkpoints) > self.max_checkpoints:
            old_path = self.saved_checkpoints.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)

        # Save latest info
        self._save_info(epoch, metrics)

        return filepath

    def load_best(self, model: torch.nn.Module,
                  trainer: Optional[Any] = None) -> Dict:
        """Load best checkpoint"""
        best_path = os.path.join(self.save_dir, 'best_checkpoint.pt')
        if not os.path.exists(best_path):
            # Find most recent
            if self.saved_checkpoints:
                best_path = self.saved_checkpoints[-1]
            else:
                raise FileNotFoundError("No checkpoints found")

        checkpoint = torch.load(best_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        if trainer and 'optimizer_state_dict' in checkpoint:
            trainer.opt.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint

    def _save_info(self, epoch: int, metrics: Dict) -> None:
        """Save training info JSON"""
        info = {
            'last_epoch': epoch,
            'best_metric': float(self.best_metric),
            'metric_name': self.metric_name,
            'num_checkpoints': len(self.saved_checkpoints),
        }

        info_path = os.path.join(self.save_dir, 'training_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)


class WandBLogger:
    """
    Optional Weights & Biases logging integration.
    
    Usage:
        logger = WandBLogger(project='zerostrike', config=config_dict)
        logger.init()
        
        for epoch in range(n_epochs):
            metrics = train_epoch()
            logger.log(metrics, epoch)
    """

    def __init__(self, project: str = 'zerostrike-drl',
                 entity: Optional[str] = None,
                 config: Optional[Dict] = None):
        self.project = project
        self.entity = entity
        self.config = config or {}
        self._initialized = False
        self._wandb = None

    def init(self, resume: bool = False, run_id: Optional[str] = None):
        """Initialize wandb run"""
        try:
            import wandb # type: ignore
            self._wandb = wandb
            self._wandb.init(
                project=self.project,
                entity=self.entity,
                config=self.config,
                resume=resume,
                id=run_id,
            )
            self._initialized = True
        except ImportError:
            print("wandb not installed. Skipping logging.")

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to wandb"""
        if self._initialized and self._wandb:
            self._wandb.log(metrics, step=step)

    def watch(self, model: torch.nn.Module):
        """Watch model gradients"""
        if self._initialized and self._wandb:
            self._wandb.watch(model)

    def finish(self):
        """End wandb run"""
        if self._initialized and self._wandb:
            self._wandb.finish()
