"""
Evaluation metrics for brain connectivity prediction
"""

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(predictions, targets):
    """
    Compute all evaluation metrics
    
    Args:
        predictions: Predicted values (numpy array)
        targets: Target values (numpy array)
    
    Returns:
        dict: Dictionary of metric values
    """
    # Flatten if matrices
    if len(predictions.shape) > 2:
        predictions = predictions.reshape(predictions.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)
    
    # Convert to numpy if tensor
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    
    metrics = {}
    
    # MSE
    metrics['mse'] = mean_squared_error(targets.flatten(), predictions.flatten())
    
    # MAE
    metrics['mae'] = mean_absolute_error(targets.flatten(), predictions.flatten())
    
    # R²
    metrics['r2'] = r2_score(targets.flatten(), predictions.flatten())
    
    # Pearson correlation
    try:
        corr, _ = pearsonr(targets.flatten(), predictions.flatten())
        metrics['pearson_correlation'] = corr
    except:
        metrics['pearson_correlation'] = 0.0
    
    # RMSE
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    return metrics


def connectivity_metrics(pred_connectivity, true_connectivity):
    """
    Compute connectivity-specific metrics
    
    Args:
        pred_connectivity: (num_nodes, num_nodes)
        true_connectivity: (num_nodes, num_nodes)
    
    Returns:
        dict: Connectivity metrics
    """
    metrics = {}
    
    # Edge-wise correlation
    pred_flat = pred_connectivity.flatten()
    true_flat = true_connectivity.flatten()
    
    corr, _ = pearsonr(pred_flat, true_flat)
    metrics['edge_correlation'] = corr
    
    # Top-k edge accuracy
    k = int(0.1 * len(pred_flat))  # Top 10%
    pred_topk = np.argsort(pred_flat)[-k:]
    true_topk = np.argsort(true_flat)[-k:]
    overlap = len(set(pred_topk) & set(true_topk))
    metrics['top10_edge_accuracy'] = overlap / k
    
    # Frobenius norm distance
    metrics['frobenius_distance'] = np.linalg.norm(pred_connectivity - true_connectivity, 'fro')
    
    return metrics


def graph_topology_metrics(pred_connectivity, true_connectivity, threshold=0.1):
    """
    Compare graph topological properties
    
    Args:
        pred_connectivity: Predicted connectivity matrix
        true_connectivity: True connectivity matrix
        threshold: Threshold for binarization
    
    Returns:
        dict: Topology metrics
    """
    import networkx as nx
    
    # Binarize
    pred_binary = (np.abs(pred_connectivity) > threshold).astype(int)
    true_binary = (np.abs(true_connectivity) > threshold).astype(int)
    
    # Create graphs
    G_pred = nx.from_numpy_array(pred_binary)
    G_true = nx.from_numpy_array(true_binary)
    
    metrics = {}
    
    # Degree distribution correlation
    deg_pred = np.array([d for n, d in G_pred.degree()])
    deg_true = np.array([d for n, d in G_true.degree()])
    metrics['degree_correlation'], _ = pearsonr(deg_pred, deg_true)
    
    # Clustering coefficient
    metrics['clustering_pred'] = nx.average_clustering(G_pred)
    metrics['clustering_true'] = nx.average_clustering(G_true)
    metrics['clustering_diff'] = abs(metrics['clustering_pred'] - metrics['clustering_true'])
    
    # Number of edges
    metrics['num_edges_pred'] = G_pred.number_of_edges()
    metrics['num_edges_true'] = G_true.number_of_edges()
    metrics['edge_count_diff'] = abs(metrics['num_edges_pred'] - metrics['num_edges_true'])
    
    return metrics
