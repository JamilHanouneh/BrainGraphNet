"""
Graph Builder - Convert connectivity matrices to graph structures
"""

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import networkx as nx


class GraphBuilder:
    """Build temporal graph sequences from connectivity matrices"""
    
    def __init__(self, config):
        self.config = config
        self.graph_config = config['data']['graph']
        self.threshold_method = self.graph_config['threshold_method']
        self.threshold_value = self.graph_config['threshold_value']
        self.min_edge_weight = self.graph_config['min_edge_weight']
        self.weighted = self.graph_config['weighted']
        self.num_regions = config['data']['atlas']['num_regions']
    
    def build_dataset(self, connectivity_data):
        """
        Build complete dataset of temporal graphs
        
        Args:
            connectivity_data: (num_subjects, num_timepoints, num_regions, num_regions)
        
        Returns:
            List of temporal graph sequences
        """
        print("Building temporal graphs...")
        graph_sequences = []
        
        for subject_idx in tqdm(range(len(connectivity_data)), desc="Processing subjects"):
            subject_connectivity = connectivity_data[subject_idx]
            graphs = self.build_temporal_graphs(subject_connectivity)
            graph_sequences.append(graphs)
        
        return graph_sequences
    
    def build_temporal_graphs(self, connectivity_sequence):
        """
        Build temporal sequence of graphs from connectivity matrices
        
        Args:
            connectivity_sequence: (num_timepoints, num_regions, num_regions)
        
        Returns:
            List of PyTorch Geometric Data objects
        """
        graphs = []
        num_timepoints = len(connectivity_sequence)
        
        for t in range(num_timepoints):
            connectivity_matrix = connectivity_sequence[t]
            graph = self.connectivity_to_graph(connectivity_matrix, timepoint=t)
            graphs.append(graph)
        
        return graphs
    
    def connectivity_to_graph(self, connectivity_matrix, timepoint=0):
        """
        Convert single connectivity matrix to PyTorch Geometric graph
        
        Args:
            connectivity_matrix: (num_regions, num_regions)
            timepoint: Current timepoint index
        
        Returns:
            PyTorch Geometric Data object
        """
        # Apply thresholding
        thresholded = self._apply_threshold(connectivity_matrix)
        
        # Build edge index and edge attributes
        edge_index, edge_attr = self._build_edges(thresholded)
        
        # Create node features (can be extended with regional properties)
        num_nodes = connectivity_matrix.shape[0]
        node_features = self._create_node_features(connectivity_matrix, num_nodes)
        
        # Create PyTorch Geometric Data object
        graph = Data(
            x=torch.FloatTensor(node_features),
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_attr),
            num_nodes=num_nodes,
            timepoint=timepoint,
            connectivity=torch.FloatTensor(connectivity_matrix)
        )
        
        return graph
    
    def _apply_threshold(self, connectivity_matrix):
        """Apply thresholding to connectivity matrix"""
        if self.threshold_method == 'none':
            return connectivity_matrix
        
        # Work with absolute values for thresholding
        abs_connectivity = np.abs(connectivity_matrix)
        
        if self.threshold_method == 'absolute':
            # Keep connections above absolute threshold
            mask = abs_connectivity >= self.threshold_value
        
        elif self.threshold_method == 'proportional':
            # Keep top X% of connections
            num_edges = int(self.threshold_value * connectivity_matrix.size)
            threshold = np.partition(abs_connectivity.flatten(), -num_edges)[-num_edges]
            mask = abs_connectivity >= threshold
        
        else:
            raise ValueError(f"Unknown threshold method: {self.threshold_method}")
        
        # Apply mask
        thresholded = connectivity_matrix.copy()
        thresholded[~mask] = 0
        
        # Remove self-connections
        np.fill_diagonal(thresholded, 0)
        
        return thresholded
    
    def _build_edges(self, connectivity_matrix):
        """Build edge_index and edge_attr from connectivity matrix"""
        # Find edges (non-zero connections)
        edge_mask = np.abs(connectivity_matrix) >= self.min_edge_weight
        sources, targets = np.where(edge_mask)
        
        # Build edge index (2 x num_edges)
        edge_index = np.stack([sources, targets], axis=0)
        
        # Build edge attributes (edge weights)
        edge_weights = connectivity_matrix[sources, targets]
        
        if not self.weighted:
            # Convert to binary edges
            edge_weights = np.ones_like(edge_weights)
        
        edge_attr = edge_weights.reshape(-1, 1)
        
        return edge_index, edge_attr
    
    def _create_node_features(self, connectivity_matrix, num_nodes):
        """
        Create node features from connectivity matrix
        Can be extended with:
        - Regional brain properties (volume, coordinates)
        - Degree centrality
        - Clinical covariates (age, sex, diagnosis)
        """
        # Basic features: node degree (sum of connections)
        node_degree = np.abs(connectivity_matrix).sum(axis=1, keepdims=True)
        
        # Normalize
        node_degree = node_degree / (node_degree.max() + 1e-8)
        
        # Can add more features here
        node_features = node_degree
        
        return node_features
    
    def graph_to_networkx(self, graph):
        """Convert PyTorch Geometric graph to NetworkX (for visualization)"""
        G = nx.Graph()
        
        num_nodes = graph.num_nodes
        G.add_nodes_from(range(num_nodes))
        
        edge_index = graph.edge_index.numpy()
        edge_attr = graph.edge_attr.numpy().flatten()
        
        for i in range(edge_index.shape[1]):
            source = edge_index[0, i]
            target = edge_index[1, i]
            weight = edge_attr[i]
            G.add_edge(source, target, weight=weight)
        
        return G
