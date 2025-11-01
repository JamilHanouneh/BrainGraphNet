"""
DataLoader utilities for temporal brain graphs
"""

import torch
from torch.utils.data import DataLoader, random_split
import numpy as np


def create_dataloaders(dataset, config, logger):
    """
    Create train/val/test dataloaders with proper splitting
    
    Args:
        dataset: TemporalBrainGraphDataset
        config: Configuration dictionary
        logger: Logger instance
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Get split ratios
    train_ratio = config['training']['train_ratio']
    val_ratio = config['training']['val_ratio']
    test_ratio = config['training']['test_ratio']
    
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    # Compute split sizes
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    logger.info(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Split dataset
    generator = torch.Generator().manual_seed(config['system']['seed'])
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # Create dataloaders
    batch_size = config['training']['batch_size']
    num_workers = config['system']['num_workers']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate,
        pin_memory=False  # Set to True if using CUDA
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader


def custom_collate(batch):
    """
    Custom collate function for temporal graphs
    Handles variable-length sequences
    """
    # Stack graphs by timepoint
    num_timepoints = batch[0].num_timepoints
    
    batched_sequences = []
    for t in range(num_timepoints):
        # Collect graphs at timepoint t
        graphs_t = [item.graphs[t] for item in batch]
        # Simple batching - store as list
        batched_sequences.append(graphs_t)
    
    # Collect targets and labels
    if torch.is_tensor(batch[0].target):
        targets = torch.stack([item.target for item in batch])
    else:
        targets = [item.target for item in batch]
    
    labels = torch.stack([item.label for item in batch])
    
    # Create batch object
    return BatchedTemporalGraphs(
        sequences=batched_sequences,
        targets=targets,
        labels=labels
    )


class BatchedTemporalGraphs:
    """Container for batched temporal graph sequences"""
    
    def __init__(self, sequences, targets, labels):
        self.sequences = sequences  # List of lists of graphs
        self.targets = targets
        self.labels = labels
        self.num_timepoints = len(sequences)
        self.batch_size = len(sequences[0])
    
    def to(self, device):
        """Move all tensors to device"""
        # Move graphs
        self.sequences = [
            [g.to(device) for g in timepoint_graphs]
            for timepoint_graphs in self.sequences
        ]
        
        # Move targets and labels
        if torch.is_tensor(self.targets):
            self.targets = self.targets.to(device)
        self.labels = self.labels.to(device)
        
        return self
    
    def __repr__(self):
        return f"BatchedTemporalGraphs(batch_size={self.batch_size}, num_timepoints={self.num_timepoints})"
