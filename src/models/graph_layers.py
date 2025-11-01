"""
Graph Neural Network Layers
Implements GCN, GAT, and custom graph convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import math


class GraphConvolution(nn.Module):
    """
    Simple Graph Convolutional Layer
    Based on Kipf & Welling (2017)
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Node features (num_nodes, in_features)
            edge_index: Edge connectivity (2, num_edges)
            edge_weight: Edge weights (num_edges,)
        """
        # Linear transformation
        support = torch.mm(x, self.weight)
        
        # Aggregate neighbors
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Apply edge weights if provided
        if edge_weight is not None:
            edge_weight = edge_weight.squeeze()
            norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        else:
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Sparse matrix multiplication
        output = torch.zeros_like(support)
        for i in range(len(row)):
            output[row[i]] += norm[i] * support[col[i]]
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class GCNLayer(nn.Module):
    """
    GCN Layer with activation and dropout
    Uses PyTorch Geometric's optimized GCNConv
    """
    
    def __init__(self, in_dim, out_dim, dropout=0.0, activation='relu'):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class TemporalGCNLayer(nn.Module):
    """
    Temporal GCN Layer with recurrent connections
    Combines GCN with GRU for temporal modeling
    """
    
    def __init__(self, in_dim, hidden_dim, dropout=0.0):
        super(TemporalGCNLayer, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Spatial convolution
        self.gcn = GCNLayer(in_dim, hidden_dim, dropout)
        
        # Temporal recurrence
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
    
    def forward(self, x, edge_index, edge_weight=None, hidden_state=None):
        """
        Args:
            x: Node features (num_nodes, in_dim)
            edge_index: Edge connectivity
            edge_weight: Edge weights
            hidden_state: Previous hidden state (num_nodes, hidden_dim)
        
        Returns:
            output: Updated node features
            hidden_state: New hidden state
        """
        # Spatial processing
        spatial_out = self.gcn(x, edge_index, edge_weight)
        
        # Temporal processing
        if hidden_state is None:
            hidden_state = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        
        hidden_state = self.gru(spatial_out, hidden_state)
        
        return hidden_state, hidden_state


class MultiHeadGATLayer(nn.Module):
    """
    Multi-head Graph Attention Layer
    Optional alternative to GCN
    """
    
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.0):
        super(MultiHeadGATLayer, self).__init__()
        self.gat = GATConv(in_dim, out_dim, heads=num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        x = self.dropout(x)
        return x
