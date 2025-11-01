"""
Helper utility functions
"""

import torch
import numpy as np
import random
from pathlib import Path


def set_seed(seed=42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, val_loss, save_path, config=None):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        val_loss: Validation loss
        save_path: Path to save checkpoint
        config: Configuration dictionary (optional)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    
    if config is not None:
        checkpoint['config'] = config
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, save_path)


def load_checkpoint(model, checkpoint_path, device, optimizer=None):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint
        device: Device to load to
        optimizer: Optimizer (optional)
    
    Returns:
        epoch: Epoch number from checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    
    return epoch


def count_parameters(model):
    """
    Count number of trainable parameters
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(device_name='auto'):
    """
    Get PyTorch device
    
    Args:
        device_name: 'auto', 'cpu', or 'cuda'
    
    Returns:
        torch.device
    """
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device(device_name)


def ensure_dir(path):
    """
    Ensure directory exists
    
    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def format_time(seconds):
    """
    Format seconds to human-readable time
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def compute_memory_usage():
    """
    Compute current memory usage
    
    Returns:
        Memory usage in MB
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    return memory_mb
