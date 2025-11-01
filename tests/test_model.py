"""
Unit tests for models
"""

import unittest
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.evolve_gcn import EvolveGCN
from src.models.graph_layers import GCNLayer
from src.utils.config_parser import load_config


class TestModels(unittest.TestCase):
    """Test neural network models"""
    
    def setUp(self):
        """Setup"""
        self.config = load_config('config.yaml')
        self.device = torch.device('cpu')
    
    def test_evolve_gcn_creation(self):
        """Test EvolveGCN model creation"""
        model = EvolveGCN(self.config)
        self.assertIsNotNone(model)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(num_params, 0)
    
    def test_gcn_layer(self):
        """Test GCN layer"""
        layer = GCNLayer(in_dim=10, out_dim=20)
        
        # Test forward pass
        x = torch.randn(5, 10)  # 5 nodes, 10 features
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        
        output = layer(x, edge_index)
        self.assertEqual(output.shape, (5, 20))


if __name__ == '__main__':
    unittest.main()
