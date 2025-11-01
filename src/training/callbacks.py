"""
Training callbacks for early stopping, checkpointing, etc.
"""

import torch
from pathlib import Path
import numpy as np


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving
    """
    
    def __init__(self, patience=10, min_delta=0.001, logger=None):
        self.patience = patience
        self.min_delta = min_delta
        self.logger = logger
        
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def on_epoch_end(self, epoch, model, train_loss, val_loss, val_metrics):
        """Called at the end of each epoch"""
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            # Improvement
            self.best_loss = val_loss
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.logger:
                self.logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                if self.logger:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                return 'stop'
        
        return None


class ModelCheckpoint:
    """
    Save model checkpoints during training
    """
    
    def __init__(self, save_dir, save_best=True, save_freq=10, max_to_keep=3, logger=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = save_best
        self.save_freq = save_freq
        self.max_to_keep = max_to_keep
        self.logger = logger
        
        self.best_loss = float('inf')
        self.saved_checkpoints = []
    
    def on_epoch_end(self, epoch, model, train_loss, val_loss, val_metrics):
        """Save checkpoint"""
        # Save best model
        if self.save_best and val_loss < self.best_loss:
            self.best_loss = val_loss
            checkpoint_path = self.save_dir / 'best_model.pth'
            self._save_checkpoint(model, checkpoint_path, epoch, val_loss, val_metrics)
            if self.logger:
                self.logger.info(f"Saved best model to {checkpoint_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % self.save_freq == 0:
            checkpoint_path = self.save_dir / f'checkpoint_epoch{epoch+1}.pth'
            self._save_checkpoint(model, checkpoint_path, epoch, val_loss, val_metrics)
            self.saved_checkpoints.append(checkpoint_path)
            
            # Remove old checkpoints
            if len(self.saved_checkpoints) > self.max_to_keep:
                old_checkpoint = self.saved_checkpoints.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
            
            if self.logger:
                self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        return None
    
    def _save_checkpoint(self, model, path, epoch, val_loss, val_metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'config': model.config
        }
        torch.save(checkpoint, path)


class LearningRateSchedulerCallback:
    """
    Custom learning rate scheduling callback
    """
    
    def __init__(self, optimizer, schedule_type='cosine', logger=None):
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.logger = logger
        self.initial_lr = optimizer.param_groups[0]['lr']
    
    def on_epoch_end(self, epoch, model, train_loss, val_loss, val_metrics):
        """Adjust learning rate"""
        if self.schedule_type == 'cosine':
            lr = self.initial_lr * (0.5 + 0.5 * np.cos(np.pi * epoch / 100))
        elif self.schedule_type == 'exponential':
            lr = self.initial_lr * (0.95 ** epoch)
        else:
            return None
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        if self.logger and epoch % 10 == 0:
            self.logger.info(f"Learning rate adjusted to {lr:.6f}")
        
        return None
