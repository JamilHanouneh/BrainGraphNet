"""
BrainGraphNet - Inference Script
Run predictions on new connectivity data
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np

sys.path.append(str(Path(__file__).parent / 'src'))

from src.utils.config_parser import load_config
from src.utils.helpers import load_checkpoint
from src.data.graph_builder import GraphBuilder
from src.models.evolve_gcn import EvolveGCN
from src.visualization.connectivity_plots import plot_prediction_comparison


def parse_args():
    parser = argparse.ArgumentParser(description='BrainGraphNet Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to connectivity data (.npy file)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--output', type=str, default='outputs/predictions/inference.npy',
                        help='Output file for predictions')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    device = torch.device(config['system']['device'])
    
    print("="*60)
    print("BrainGraphNet Inference")
    print("="*60)
    
    # Load connectivity data
    print(f"Loading data from {args.data}...")
    connectivity = np.load(args.data)
    print(f"Data shape: {connectivity.shape}")
    
    # Expected shape: (num_timepoints, num_regions, num_regions)
    if len(connectivity.shape) != 3:
        raise ValueError(f"Expected 3D array, got shape {connectivity.shape}")
    
    # Build graphs
    print("Building graphs...")
    graph_builder = GraphBuilder(config)
    graphs = graph_builder.build_temporal_graphs(connectivity)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = EvolveGCN(config).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    
    # Predict
    print("Running inference...")
    with torch.no_grad():
        predictions = model(graphs)
        predictions = predictions.cpu().numpy()
    
    print(f"Prediction shape: {predictions.shape}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, predictions)
    print(f"✅ Predictions saved to {output_path}")
    
    # Visualize
    plot_path = output_path.parent / (output_path.stem + '_plot.png')
    plot_prediction_comparison(connectivity[-1], predictions, save_path=plot_path)
    print(f"✅ Visualization saved to {plot_path}")


if __name__ == "__main__":
    main()
