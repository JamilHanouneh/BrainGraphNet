"""
EvolveGCN: Evolving Graph Convolutional Networks
Based on Pareja et al. (2020) - AAAI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_layers import GraphConvolution


class EvolveGCN(nn.Module):
    """EvolveGCN Model for Temporal Graph Learning"""
    
    def __init__(self, config):
        super(EvolveGCN, self).__init__()
        self.config = config
        model_config = config['model']
        arch_config = model_config['architecture']
        evolve_config = model_config['evolve_gcn']
        
        self.input_dim = arch_config['input_dim']
        self.hidden_dim = arch_config['hidden_dim']
        self.output_dim = arch_config['output_dim']
        self.num_layers = arch_config['num_layers']
        self.dropout = arch_config['dropout']
        self.rnn_type = evolve_config['rnn_type']
        self.variant = evolve_config['variant']
        
        self.num_regions = config['data']['atlas']['num_regions']
        self.task_type = model_config['task']['type']
        
        # Build model
        self._build_model()
    
    def _build_model(self):
        """Build EvolveGCN architecture"""
        # Evolving GCN layers
        self.gcn_layers = nn.ModuleList()
        self.rnn_cells = nn.ModuleList()
        
        # Layer dimensions
        layer_dims = [self.input_dim]
        for i in range(self.num_layers - 1):
            layer_dims.append(self.hidden_dim)
        layer_dims.append(self.output_dim)
        
        # Create layers
        for i in range(self.num_layers):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            
            # GCN layer
            gcn = GraphConvolution(in_dim, out_dim)
            self.gcn_layers.append(gcn)
            
            # RNN to evolve GCN weights
            weight_size = in_dim * out_dim
            if self.rnn_type == 'GRU':
                rnn = nn.GRUCell(weight_size, weight_size)
            else:  # LSTM
                rnn = nn.LSTMCell(weight_size, weight_size)
            self.rnn_cells.append(rnn)
        
        # Output layers based on task
        if self.task_type == 'connectivity_prediction':
            # Use the actual output_dim from last layer
            self.fc_out = nn.Linear(self.output_dim, self.output_dim)
        else:  # phenotype_classification
            self.fc_out = nn.Sequential(
                nn.Linear(self.output_dim * self.num_regions, 128),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
    def forward(self, batch):
        """Forward pass through temporal graph sequence"""
        device = next(self.parameters()).device
        batch_size = batch.batch_size
        num_timepoints = batch.num_timepoints

        # Initialize RNN hidden states for each batch item
        hidden_states = []
        for layer_idx in range(self.num_layers):
            gcn_layer = self.gcn_layers[layer_idx]
            weight_size = gcn_layer.weight.numel()
            h = torch.zeros(batch_size, weight_size, device=device)
            hidden_states.append(h)

        # Process temporal sequence
        all_embeddings = []

        for t in range(num_timepoints):
            # Get graphs at timepoint t
            graphs_t = batch.sequences[t]

            # Process each graph in batch
            batch_embeddings = []
            new_hidden_states = [[] for _ in range(self.num_layers)]  # Store new states separately

            for b_idx, graph in enumerate(graphs_t):
                x = graph.x.to(device)
                edge_index = graph.edge_index.to(device)
                edge_weight = graph.edge_attr.squeeze().to(device) if hasattr(graph, 'edge_attr') and graph.edge_attr is not None else None

                # Forward through evolved GCN layers
                for layer_idx in range(self.num_layers):
                    # Get current hidden state (don't modify original)
                    h_current = hidden_states[layer_idx][b_idx:b_idx+1].clone()

                    # Evolve weights
                    h_new = self.rnn_cells[layer_idx](h_current)
                    new_hidden_states[layer_idx].append(h_new)

                    # Reshape to weight matrix
                    gcn_layer = self.gcn_layers[layer_idx]
                    in_dim = gcn_layer.in_features
                    out_dim = gcn_layer.out_features
                    evolved_weight = h_new.view(in_dim, out_dim)

                    # Apply GCN with evolved weights
                    x = self._apply_gcn(x, edge_index, edge_weight, evolved_weight, gcn_layer.bias)

                    # Apply activation and dropout (except last layer)
                    if layer_idx < self.num_layers - 1:
                        x = F.relu(x)
                        x = self.dropout_layer(x)

                batch_embeddings.append(x)

            # Update hidden states after processing all graphs in batch
            for layer_idx in range(self.num_layers):
                hidden_states[layer_idx] = torch.cat(new_hidden_states[layer_idx], dim=0)

            # Stack batch embeddings
            batch_embeddings_tensor = torch.stack(batch_embeddings)
            all_embeddings.append(batch_embeddings_tensor)

        # Use final timestep embeddings for prediction
        final_embeddings = all_embeddings[-1]  # (batch_size, num_nodes, output_dim)

        # Generate predictions
        if self.task_type == 'connectivity_prediction':
            predictions = self._predict_connectivity(final_embeddings)
        else:  # phenotype_classification
            global_embed = final_embeddings.mean(dim=1)  # (batch_size, output_dim)
            predictions = self.fc_out(global_embed.view(batch_size, -1))

        return predictions

    # def forward(self, batch):
    #     """Forward pass through temporal graph sequence"""
    #     device = next(self.parameters()).device
    #     batch_size = batch.batch_size
    #     num_timepoints = batch.num_timepoints
        
    #     # Initialize RNN hidden states for each batch item
    #     hidden_states = []
    #     for layer_idx in range(self.num_layers):
    #         gcn_layer = self.gcn_layers[layer_idx]
    #         weight_size = gcn_layer.weight.numel()
    #         h = torch.zeros(batch_size, weight_size, device=device)
    #         hidden_states.append(h)
        
    #     # Process temporal sequence
    #     all_embeddings = []
        
    #     for t in range(num_timepoints):
    #         # Get graphs at timepoint t
    #         graphs_t = batch.sequences[t]
            
    #         # Process each graph in batch
    #         batch_embeddings = []
    #         for b_idx, graph in enumerate(graphs_t):
    #             x = graph.x.to(device)
    #             edge_index = graph.edge_index.to(device)
    #             edge_weight = graph.edge_attr.squeeze().to(device) if hasattr(graph, 'edge_attr') and graph.edge_attr is not None else None
                
    #             # Forward through evolved GCN layers
    #             for layer_idx in range(self.num_layers):
    #                 # Evolve weights
    #                 h = hidden_states[layer_idx][b_idx:b_idx+1]
    #                 h_new = self.rnn_cells[layer_idx](h)
    #                 hidden_states[layer_idx][b_idx:b_idx+1] = h_new
                    
    #                 # Reshape to weight matrix
    #                 gcn_layer = self.gcn_layers[layer_idx]
    #                 in_dim = gcn_layer.in_features
    #                 out_dim = gcn_layer.out_features
    #                 evolved_weight = h_new.view(in_dim, out_dim)
                    
    #                 # Apply GCN with evolved weights
    #                 x = self._apply_gcn(x, edge_index, edge_weight, evolved_weight, gcn_layer.bias)
                    
    #                 # Apply activation and dropout (except last layer)
    #                 if layer_idx < self.num_layers - 1:
    #                     x = F.relu(x)
    #                     x = self.dropout_layer(x)
                
    #             batch_embeddings.append(x)
            
    #         # Stack batch embeddings
    #         batch_embeddings_tensor = torch.stack(batch_embeddings)
    #         all_embeddings.append(batch_embeddings_tensor)
        
    #     # Use final timestep embeddings for prediction
    #     final_embeddings = all_embeddings[-1]  # (batch_size, num_nodes, output_dim)
        
    #     # Generate predictions
    #     if self.task_type == 'connectivity_prediction':
    #         predictions = self._predict_connectivity(final_embeddings)
    #     else:  # phenotype_classification
    #         global_embed = final_embeddings.mean(dim=1)  # (batch_size, output_dim)
    #         predictions = self.fc_out(global_embed.view(batch_size, -1))
        
    #     return predictions
    
    def _apply_gcn(self, x, edge_index, edge_weight, weight, bias):
        """Apply graph convolution with given weights"""
        # Linear transformation
        support = torch.mm(x, weight)
        
        # Aggregate neighbors
        row, col = edge_index
        num_nodes = x.size(0)
        deg = torch.bincount(row, minlength=num_nodes).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        if edge_weight is not None and edge_weight.numel() > 0:
            norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        else:
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Message passing
        output = torch.zeros_like(support)
        for i in range(len(row)):
            output[row[i]] += norm[i] * support[col[i]]
        
        if bias is not None:
            output = output + bias
        
        return output
    
    def _predict_connectivity(self, node_embeddings):
        """
        Predict connectivity matrix from node embeddings
        
        Args:
            node_embeddings: (batch_size, num_nodes, embedding_dim)
        
        Returns:
            connectivity: (batch_size, num_nodes, num_nodes)
        """
        batch_size = node_embeddings.size(0)
        num_nodes = node_embeddings.size(1)
        
        # Apply output transformation to each node
        # Transform: (batch_size, num_nodes, output_dim) -> (batch_size, num_nodes, output_dim)
        transformed = self.fc_out(node_embeddings)
        
        # Compute pairwise similarity (connectivity)
        # Use batch matrix multiplication
        connectivity = torch.bmm(transformed, transformed.transpose(1, 2))
        
        # Normalize to [-1, 1] range (correlation-like)
        connectivity = torch.tanh(connectivity)
        
        return connectivity
    
    @staticmethod
    def load_from_checkpoint(checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        model = EvolveGCN(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
