"""
Trainer class for temporal graph neural networks
Handles training loop, validation, and logging
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import time
from pathlib import Path

from ..models.loss import CombinedConnectivityLoss
from ..models.metrics import compute_metrics
from .optimizer import create_optimizer, create_scheduler


class Trainer:
    """
    Trainer for temporal brain graph models
    """
    
    def __init__(self, model, train_loader, val_loader, config, device, logger, callbacks=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger
        self.callbacks = callbacks or []
        
        self.train_config = config['training']
        self.num_epochs = self.train_config['num_epochs']
        
        # Loss function
        self.criterion = CombinedConnectivityLoss(
            lambda_corr=0.5,
            lambda_tc=0.1
        )
        
        # Optimizer and scheduler
        self.optimizer = create_optimizer(model, config)
        self.scheduler = create_scheduler(self.optimizer, config)
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        
        # TensorBoard (optional)
        self.use_tensorboard = config['system']['logging'].get('tensorboard', False)
        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = Path(config['system']['paths']['logs']) / 'tensorboard'
            self.writer = SummaryWriter(log_dir)
    
    def train(self, start_epoch=0):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        for epoch in range(start_epoch, self.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train one epoch
            train_loss, train_metrics = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Track losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Log
            epoch_time = time.time() - epoch_start_time
            self._log_epoch(epoch, train_loss, val_loss, train_metrics, val_metrics, epoch_time)
            
            # TensorBoard logging
            if self.use_tensorboard:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                for metric_name, value in val_metrics.items():
                    self.writer.add_scalar(f'Metrics/{metric_name}', value, epoch)
            
            # Track best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.logger.info(f"New best validation loss: {val_loss:.4f}")
            
            # Callbacks
            stop_training = False
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    result = callback.on_epoch_end(
                        epoch=epoch,
                        model=self.model,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        val_metrics=val_metrics
                    )
                    if result == 'stop':
                        stop_training = True
            
            if stop_training:
                self.logger.info("Early stopping triggered")
                break
        
        if self.use_tensorboard:
            self.writer.close()
        
        self.logger.info("Training completed!")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch)
            
            # Compute loss
            targets = batch.targets
            loss, loss_components = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (optional)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track
            total_loss += loss.item()
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(targets.detach().cpu())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        
        predictions_cat = torch.cat(all_predictions, dim=0).numpy()
        targets_cat = torch.cat(all_targets, dim=0).numpy()
        metrics = compute_metrics(predictions_cat, targets_cat)
        
        return avg_loss, metrics
    
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                predictions = self.model(batch)
                targets = batch.targets
                
                # Compute loss
                loss, _ = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        
        predictions_cat = torch.cat(all_predictions, dim=0).numpy()
        targets_cat = torch.cat(all_targets, dim=0).numpy()
        metrics = compute_metrics(predictions_cat, targets_cat)
        
        return avg_loss, metrics
    
    def evaluate(self, test_loader):
        """Evaluate on test set"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                batch = batch.to(self.device)
                
                predictions = self.model(batch)
                all_predictions.append(predictions.cpu())
                all_targets.append(batch.targets.cpu())
        
        predictions_cat = torch.cat(all_predictions, dim=0).numpy()
        targets_cat = torch.cat(all_targets, dim=0).numpy()
        metrics = compute_metrics(predictions_cat, targets_cat)
        
        return metrics
    
    def _log_epoch(self, epoch, train_loss, val_loss, train_metrics, val_metrics, epoch_time):
        """Log epoch information"""
        self.logger.info(f"\nEpoch {epoch+1}/{self.num_epochs} - {epoch_time:.1f}s")
        self.logger.info(f"  Train Loss: {train_loss:.4f}")
        self.logger.info(f"  Val Loss:   {val_loss:.4f}")
        
        # Log key metrics
        for metric_name in ['mse', 'mae', 'r2', 'pearson_correlation']:
            if metric_name in val_metrics:
                train_val = train_metrics.get(metric_name, 0)
                val_val = val_metrics.get(metric_name, 0)
                self.logger.info(f"  {metric_name}: Train={train_val:.4f}, Val={val_val:.4f}")
