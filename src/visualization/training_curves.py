"""
Training curve visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save figure
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_metrics(metrics_history, save_path=None):
    """
    Plot multiple metrics over epochs
    
    Args:
        metrics_history: Dict of metric_name -> list of values
        save_path: Path to save figure
    """
    num_metrics = len(metrics_history)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (metric_name, values) in enumerate(metrics_history.items()):
        if idx >= 4:  # Max 4 subplots
            break
        
        epochs = range(1, len(values) + 1)
        axes[idx].plot(epochs, values, marker='o', linewidth=2)
        axes[idx].set_xlabel('Epoch', fontsize=11)
        axes[idx].set_ylabel(metric_name.upper(), fontsize=11)
        axes[idx].set_title(f'{metric_name.replace("_", " ").title()} Evolution',
                           fontsize=12, fontweight='bold')
        axes[idx].grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_metrics, 4):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_learning_rate_schedule(learning_rates, save_path=None):
    """
    Plot learning rate schedule
    
    Args:
        learning_rates: List of learning rates per epoch
        save_path: Path to save figure
    """
    epochs = range(1, len(learning_rates) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, learning_rates, marker='o', linewidth=2, color='purple')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix (for classification tasks)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_prediction_scatter(y_true, y_pred, save_path=None):
    """
    Scatter plot of true vs predicted values
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 8))
    
    plt.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.5, s=10)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
            label='Perfect Prediction')
    
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title('Prediction Scatter Plot', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
