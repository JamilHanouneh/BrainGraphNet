"""
Unit tests for data processing
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.synthetic_generator import SyntheticBrainDataGenerator
from src.data.graph_builder import GraphBuilder
from src.utils.config_parser import load_config


class TestDataProcessing(unittest.TestCase):
    """Test data processing pipeline"""
    
    def setUp(self):
        """Setup test config"""
        self.config = load_config('config.yaml')
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generator"""
        generator = SyntheticBrainDataGenerator(self.config)
        connectivity, labels = generator.generate()
        
        # Check shapes
        self.assertEqual(len(connectivity.shape), 4)
        self.assertEqual(connectivity.shape[2], connectivity.shape[3])
        self.assertEqual(len(labels), connectivity.shape[0])
        
        # Check values
        self.assertTrue(np.all(connectivity >= -1))
        self.assertTrue(np.all(connectivity <= 1))
    
    def test_graph_building(self):
        """Test graph construction"""
        # Generate test data
        connectivity = np.random.randn(2, 3, 10, 10)
        connectivity = (connectivity + connectivity.transpose(0, 1, 3, 2)) / 2
        
        builder = GraphBuilder(self.config)
        graphs = builder.build_dataset(connectivity)
        
        self.assertEqual(len(graphs), 2)  # 2 subjects
        self.assertEqual(len(graphs[0]), 3)  # 3 timepoints


if __name__ == '__main__':
    unittest.main()
