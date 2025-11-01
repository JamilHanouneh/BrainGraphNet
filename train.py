"""
BrainGraphNet - Training Script
Train temporal GNN on brain connectivity data
"""

import argparse
import sys
from pathlib import Path
import torch
import yaml
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.utils.logger import setup_logger
from src.utils.config_parser import load_config
from src.utils.helpers import set_seed, save_checkpoint, load_checkpoint
from src.data.synthetic_generator import SyntheticBrainDataGenerator
from src.data.connectivity_loader import ConnectivityLoader
from src.data.graph_builder import GraphBuilder
from src.data.dataset import TemporalBrainGraphDataset
from src.data.dataloader import create_dataloaders
from src.models.evolve_gcn import EvolveGCN
from src.models.temporal_gcn import TemporalGCN
from src.training.trainer import Trainer
from src.training.callbacks import EarlyStopping, ModelCheckpoint


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train BrainGraphNet')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cpu or cuda')
    return parser.parse_args()


def load_data(config, logger):
    """Load or generate brain connectivity data"""
    logger.info("Loading data...")
    
    data_source = config['data']['source']
    
    if data_source == 'synthetic':
        logger.info("Generating synthetic brain connectivity data...")
        generator = SyntheticBrainDataGenerator(config)
        connectivity_data, labels = generator.generate()
        
    elif data_source == 'hcp':
        logger.info("Loading HCP connectivity data...")
        loader = ConnectivityLoader(config)
        connectivity_data, labels = loader.load_hcp_data()
        
    elif data_source == 'custom':
        logger.info("Loading custom connectivity data...")
        loader = ConnectivityLoader(config)
        connectivity_data, labels = loader.load_custom_data()
        
    else:
        raise ValueError(f"Unknown data source: {data_source}")
    
    logger.info(f"Data shape: {connectivity_data.shape}")
    logger.info(f"Number of subjects: {connectivity_data.shape[0]}")
    logger.info(f"Number of timepoints: {connectivity_data.shape[1]}")
    logger.info(f"Number of regions: {connectivity_data.shape[2]}")
    
    return connectivity_data, labels


def build_graphs(connectivity_data, config, logger):
    """Convert connectivity matrices to graphs"""
    logger.info("Building temporal brain graphs...")
    
    graph_builder = GraphBuilder(config)
    graph_data = graph_builder.build_dataset(connectivity_data)
    
    logger.info(f"Built {len(graph_data)} temporal graph sequences")
    return graph_data


def create_model(config, logger):
    """Create temporal GNN model"""
    model_type = config['model']['type']
    
    logger.info(f"Creating {model_type} model...")
    
    if model_type == 'EvolveGCN':
        model = EvolveGCN(config)
    elif model_type == 'TemporalGCN':
        model = TemporalGCN(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    return model


def main():
    """Main training function"""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.device:
        config['system']['device'] = args.device
    
    # Setup
    logger = setup_logger(config)
    logger.info("="*60)
    logger.info("BrainGraphNet Training")
    logger.info("="*60)
    
    # Set random seed for reproducibility
    set_seed(config['system']['seed'])
    
    # Device
    device = torch.device(config['system']['device'])
    logger.info(f"Using device: {device}")
    
    # Load data
    connectivity_data, labels = load_data(config, logger)
    
    # Build graphs
    graph_data = build_graphs(connectivity_data, config, logger)
    
    # Create dataset
    dataset = TemporalBrainGraphDataset(graph_data, labels, config)
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, config, logger
    )
    
    # Create model
    model = create_model(config, logger)
    model = model.to(device)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch = load_checkpoint(model, args.resume, device)
    
    # Create callbacks
    callbacks = []
    
    if config['training']['early_stopping']['enabled']:
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta'],
            logger=logger
        )
        callbacks.append(early_stopping)
    
    checkpoint_callback = ModelCheckpoint(
        save_dir=config['system']['paths']['checkpoints'],
        save_best=config['training']['checkpoint']['save_best'],
        save_freq=config['training']['checkpoint']['save_freq'],
        logger=logger
    )
    callbacks.append(checkpoint_callback)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        logger=logger,
        callbacks=callbacks
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train(start_epoch=start_epoch)
    
    # Test on best model
    logger.info("Evaluating best model on test set...")
    best_checkpoint = config['system']['paths']['checkpoints'] + '/best_model.pth'
    if Path(best_checkpoint).exists():
        load_checkpoint(model, best_checkpoint, device)
        test_metrics = trainer.evaluate(test_loader)
        
        logger.info("Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("="*60)
    logger.info("Training Complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
