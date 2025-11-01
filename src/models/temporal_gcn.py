"""
Temporal GCN - Alternative simpler model
Uses stacked GCN + GRU layers
"""

import torch
import torch.nn as nn
from .graph_layers import TemporalGCNLayer


class TemporalGCN(nn.Module):
    """
    Temporal Graph Convolutional Network
    Simpler alternative to EvolveGCN
    """
    
    def __init__(self, config):
        super(TemporalGCN, self).__init__()
        self.config = config
        model_config = config['model']
        arch_config = model_config['architecture']
        
        self.input_dim = arch_config['input_dim']
        self.hidden_dim = arch_config['hidden_dim']
        self.output_dim = arch_config['output_dim']
        self.num_layers = arch_config['num_layers']
        self.dropout = arch_config['dropout']
        
        self.task_type = model_config['task']['type']
        self.num_regions = config['data']['atlas']['num_regions']
        
        # Build layers
        self.temporal_layers = nn.ModuleList()
        
        in_dim = self.input_dim
        for i in range(self.num_layers):
            out_dim = self.hidden_dim if i < self.num_layers - 1 else self.output_dim
            layer = TemporalGCNLayer(in_dim, out_dim, self.dropout)
            self.temporal_layers.append(layer)
            in_dim = out_dim
        
        # Output head
        if self.task_type == 'connectivity_prediction':
            self.output_layer = nn.Linear(self.output_dim, self.output_dim)
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(self.output_dim * self.num_regions, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
    
    def forward(self, batch):
        """Forward pass"""
        device = batch.sequences[0][0].x.device
        batch_size = batch.batch_size
        num_timepoints = batch.num_timepoints
        
        # Initialize hidden states
        hidden_states = [None] * self.num_layers
        
        # Process temporal sequence
        for t in range(num_timepoints):
            graphs_t = batch.sequences[t]
            
            for b_idx, graph in enumerate(graphs_t):
                x = graph.x
                edge_index = graph.edge_index
                edge_weight = graph.edge_attr.squeeze() if hasattr(graph, 'edge_attr') else None
                
                # Pass through temporal layers
                for layer_idx, layer in enumerate(self.temporal_layers):
                    x, hidden_states[layer_idx] = layer(
                        x, edge_index, edge_weight, 
                        hidden_states[layer_idx]
                    )
        
        # Generate predictions using final hidden states
        final_hidden = hidden_states[-1]
        
        if self.task_type == 'connectivity_prediction':
            # Predict connectivity
            transformed = self.output_layer(final_hidden)
            connectivity = torch.mm(transformed, transformed.t())
            connectivity = torch.tanh(connectivity)
            return connectivity.unsqueeze(0)
        else:
            # Classify
            global_feat = final_hidden.mean(dim=0).unsqueeze(0)
            return self.output_layer(global_feat.view(1, -1))
