"""
Connectivity matrix visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_connectivity_matrix(connectivity, title="Connectivity Matrix", 
                             save_path=None, cmap='RdBu_r', vmin=-1, vmax=1):
    """
    Plot connectivity matrix as heatmap
    
    Args:
        connectivity: (N, N) connectivity matrix
        title: Plot title
        save_path: Path to save figure
        cmap: Colormap
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        connectivity,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        square=True,
        cbar_kws={'label': 'Correlation'},
        xticklabels=False,
        yticklabels=False
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Brain Region', fontsize=12)
    plt.ylabel('Brain Region', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_prediction_comparison(true_connectivity, pred_connectivity, 
                               save_path=None):
    """
    Plot comparison between true and predicted connectivity
    
    Args:
        true_connectivity: Ground truth (N, N)
        pred_connectivity: Prediction (N, N)
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # True connectivity
    im1 = axes[0].imshow(true_connectivity, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title('True Connectivity', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Brain Region')
    axes[0].set_ylabel('Brain Region')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Predicted connectivity
    im2 = axes[1].imshow(pred_connectivity, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title('Predicted Connectivity', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Brain Region')
    axes[1].set_ylabel('Brain Region')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Difference
    difference = pred_connectivity - true_connectivity
    im3 = axes[2].imshow(difference, cmap='coolwarm', vmin=-0.5, vmax=0.5)
    axes[2].set_title('Difference (Pred - True)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Brain Region')
    axes[2].set_ylabel('Brain Region')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_temporal_connectivity_evolution(connectivity_sequence, save_path=None):
    """
    Plot temporal evolution of connectivity
    
    Args:
        connectivity_sequence: List of (N, N) matrices over time
        save_path: Path to save figure
    """
    num_timepoints = len(connectivity_sequence)
    
    fig, axes = plt.subplots(1, num_timepoints, figsize=(5*num_timepoints, 4))
    
    if num_timepoints == 1:
        axes = [axes]
    
    for t, conn in enumerate(connectivity_sequence):
        im = axes[t].imshow(conn, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[t].set_title(f'Timepoint {t+1}', fontsize=12, fontweight='bold')
        axes[t].set_xlabel('Region')
        axes[t].set_ylabel('Region')
        plt.colorbar(im, ax=axes[t], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_connectivity_distribution(connectivity, save_path=None):
    """
    Plot distribution of connectivity values
    
    Args:
        connectivity: (N, N) connectivity matrix
        save_path: Path to save figure
    """
    # Get upper triangle (exclude diagonal)
    triu_indices = np.triu_indices_from(connectivity, k=1)
    values = connectivity[triu_indices]
    
    plt.figure(figsize=(10, 6))
    
    # Histogram
    plt.hist(values, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Connectivity Strength', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Connectivity Values', fontsize=14, fontweight='bold')
    
    # Add statistics
    mean_val = np.mean(values)
    std_val = np.std(values)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    plt.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=2, label=f'±1 SD: {std_val:.3f}')
    plt.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=2)
    
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
