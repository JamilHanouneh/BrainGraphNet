"""
Optimizer and learning rate scheduler creation
"""

import torch.optim as optim


def create_optimizer(model, config):
    """
    Create optimizer based on config
    
    Args:
        model: PyTorch model
        config: Configuration dictionary
    
    Returns:
        optimizer: PyTorch optimizer
    """
    train_config = config['training']
    optimizer_name = train_config['optimizer'].lower()
    lr = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer, config):
    """
    Create learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary
    
    Returns:
        scheduler: Learning rate scheduler or None
    """
    train_config = config['training']
    
    if 'scheduler' not in train_config:
        return None
    
    scheduler_config = train_config['scheduler']
    scheduler_type = scheduler_config['type'].lower()
    
    if scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config['step_size'],
            gamma=scheduler_config['gamma']
        )
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config['num_epochs']
        )
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    elif scheduler_type == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95
        )
    else:
        return None
    
    return scheduler
