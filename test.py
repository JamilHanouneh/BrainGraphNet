"""
BrainGraphNet - Testing Script
Evaluate trained model on test data
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np

sys.path.append(str(Path(__file__).parent / 'src'))

from src.utils.logger import setup_logger
from src.utils.config_parser import load_config
from src.utils.helpers import load_checkpoint
from src.data.synthetic_generator import SyntheticBrainDataGenerator
from src.data.graph_builder import GraphBuilder
from src.data.dataset import TemporalBrainGraphDataset
from src.data.dataloader import create_dataloaders
from src.models.evolve_gcn import EvolveGCN
from src.models.metrics import compute_metrics
from src.visualization.connectivity_plots import plot_prediction_comparison
from src.visualization.brain_plots import plot_brain_network


def parse_args():
    parser = argparse.ArgumentParser(description='Test BrainGraphNet')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save predictions to file')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    logger = setup_logger(config)
    
    logger.info("="*60)
    logger.info("BrainGraphNet Testing")
    logger.info("="*60)
    
    device = torch.device(config['system']['device'])
    
    # Load data
    logger.info("Loading test data...")
    generator = SyntheticBrainDataGenerator(config)
    connectivity_data, labels = generator.generate()
    
    # Build graphs
    graph_builder = GraphBuilder(config)
    graph_data = graph_builder.build_dataset(connectivity_data)
    
    # Create dataset
    dataset = TemporalBrainGraphDataset(graph_data, labels, config)
    _, _, test_loader = create_dataloaders(dataset, config, logger)
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = EvolveGCN(config).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    
    # Evaluate
    logger.info("Evaluating...")
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            predictions = model(batch)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch.targets.cpu().numpy())
    
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    
    # Compute metrics
    metrics = compute_metrics(predictions, targets)
    
    logger.info("Test Results:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save predictions
    if args.save_predictions:
        save_path = Path(config['system']['paths']['predictions']) / 'test_predictions.npz'
        np.savez(save_path, predictions=predictions, targets=targets)
        logger.info(f"Predictions saved to {save_path}")
    
    # Visualizations
    if args.visualize:
        logger.info("Generating visualizations...")
        fig_dir = Path(config['system']['paths']['figures'])
        
        # Plot predictions
        plot_prediction_comparison(
            targets[0], predictions[0],
            save_path=fig_dir / 'test_prediction.png'
        )
        
        # Plot brain network
        plot_brain_network(
            predictions[0],
            save_path=fig_dir / 'test_brain_network.png'
        )
        
        logger.info(f"Visualizations saved to {fig_dir}")
    
    logger.info("Testing complete!")


if __name__ == "__main__":
    main()
