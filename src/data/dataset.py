"""
PyTorch Dataset for temporal brain graphs
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import numpy as np


class TemporalBrainGraphDataset(Dataset):
    """
    Dataset for temporal brain graph sequences
    Each sample is a sequence of graphs over time
    """
    
    def __init__(self, graph_sequences, labels, config):
        """
        Args:
            graph_sequences: List of temporal graph sequences
            labels: Array of labels for each sequence
            config: Configuration dictionary
        """
        self.graph_sequences = graph_sequences
        self.labels = torch.FloatTensor(labels)
        self.config = config
        self.task_type = config['model']['task']['type']
        self.prediction_horizon = config['model']['task']['prediction_horizon']
    
    def __len__(self):
        return len(self.graph_sequences)
    
    def __getitem__(self, idx):
        """
        Returns:
            graphs: Sequence of graphs [G_1, G_2, ..., G_T]
            target: Target for prediction (connectivity matrix or label)
            label: Subject-level label
        """
        graph_sequence = self.graph_sequences[idx]
        label = self.labels[idx]
        
        # Prepare target based on task
        if self.task_type == 'connectivity_prediction':
            # Target is the connectivity at t+k
            target_idx = min(len(graph_sequence) - 1, 
                           len(graph_sequence) - 1 + self.prediction_horizon)
            target = graph_sequence[target_idx].connectivity
            
            # Input is all graphs except the target
            input_graphs = graph_sequence[:-self.prediction_horizon] if self.prediction_horizon > 0 else graph_sequence
        
        elif self.task_type == 'phenotype_classification':
            # Target is the label
            target = label
            input_graphs = graph_sequence
        
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        
        # Package as TemporalGraphData
        temporal_data = TemporalGraphData(
            graphs=input_graphs,
            target=target,
            label=label
        )
        
        return temporal_data
    
    def get_graph_statistics(self):
        """Compute dataset statistics"""
        num_nodes = []
        num_edges = []
        avg_degree = []
        
        for seq in self.graph_sequences:
            for graph in seq:
                num_nodes.append(graph.num_nodes)
                num_edges.append(graph.edge_index.shape[1])
                avg_degree.append(graph.edge_index.shape[1] / graph.num_nodes)
        
        stats = {
            'avg_num_nodes': np.mean(num_nodes),
            'avg_num_edges': np.mean(num_edges),
            'avg_degree': np.mean(avg_degree)
        }
        return stats


class TemporalGraphData:
    """Container for temporal graph sequence"""
    
    def __init__(self, graphs, target, label):
        self.graphs = graphs
        self.target = target
        self.label = label
        self.num_timepoints = len(graphs)
    
    def to(self, device):
        """Move all data to device"""
        self.graphs = [g.to(device) if hasattr(g, 'to') else g for g in self.graphs]
        if torch.is_tensor(self.target):
            self.target = self.target.to(device)
        if torch.is_tensor(self.label):
            self.label = self.label.to(device)
        return self


def collate_temporal_graphs(batch):
    """
    Custom collate function for temporal graph sequences
    
    Args:
        batch: List of TemporalGraphData objects
    
    Returns:
        Batched temporal graphs
    """
    # Batch graphs at each timepoint separately
    num_timepoints = batch[0].num_timepoints
    
    batched_graphs_per_time = []
    for t in range(num_timepoints):
        graphs_at_t = [item.graphs[t] for item in batch]
        batched_graph = Batch.from_data_list(graphs_at_t)
        batched_graphs_per_time.append(batched_graph)
    
    # Stack targets
    targets = torch.stack([item.target for item in batch])
    labels = torch.stack([item.label for item in batch])
    
    return {
        'graphs': batched_graphs_per_time,
        'targets': targets,
        'labels': labels,
        'batch_size': len(batch)
    }
