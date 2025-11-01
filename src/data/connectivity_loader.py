"""Load connectivity matrices from various sources"""

import numpy as np
from pathlib import Path
import nibabel as nib
from tqdm import tqdm


class ConnectivityLoader:
    """Load brain connectivity matrices"""
    
    def __init__(self, config):
        self.config = config
    
    def load_hcp_data(self):
        """Load HCP connectivity matrices"""
        hcp_config = self.config['data']['hcp']
        data_path = Path(hcp_config['data_path'])
        
        if not data_path.exists():
            raise FileNotFoundError(f"HCP data not found at {data_path}")
        
        # Look for connectivity files
        conn_files = list(data_path.glob('*.txt')) + list(data_path.glob('*.pconn.nii'))
        
        if len(conn_files) == 0:
            raise FileNotFoundError(f"No connectivity files found in {data_path}")
        
        print(f"Found {len(conn_files)} connectivity files")
        
        # Load matrices
        connectivity_matrices = []
        for file in tqdm(conn_files[:hcp_config['num_subjects']], desc="Loading HCP data"):
            if file.suffix == '.txt':
                matrix = np.loadtxt(file)
            elif file.suffix == '.nii':
                matrix = nib.load(str(file)).get_fdata()
            
            connectivity_matrices.append(matrix)
        
        # Convert to temporal format (simulate multiple timepoints)
        connectivity_data = self._simulate_temporal(connectivity_matrices)
        labels = np.zeros(len(connectivity_data))  # Placeholder
        
        return connectivity_data, labels
    

    
    def load_custom_data(self):
        """Load custom connectivity data"""
        # Try multiple possible paths
        possible_paths = [
            Path(self.config['data'].get('custom_path', 'data/raw/custom')),
            Path('data/processed/HCP_connectivity'),
            Path('data/raw/custom')
        ]

        custom_path = None
        for path in possible_paths:
            if path.exists():
                custom_path = path
                break
            
        if custom_path is None:
            raise FileNotFoundError(f"No connectivity data found in any expected location")

        # Try different file names
        file_candidates = [
            custom_path / 'all_connectivity_temporal.npy',
            custom_path / 'connectivity.npy',
            custom_path / 'connectivity_data.npy'
        ]

        connectivity_data = None
        for file_path in file_candidates:
            if file_path.exists():
                connectivity_data = np.load(file_path)
                print(f"✅ Loaded data from: {file_path}")
                break
            
        if connectivity_data is None:
            raise FileNotFoundError(f"No connectivity data found in {custom_path}")

        # Load labels if available
        label_file = custom_path / 'labels.npy'
        if label_file.exists():
            labels = np.load(label_file)
        else:
            # Create dummy labels
            labels = np.zeros(len(connectivity_data))

        return connectivity_data, labels
    
    def _simulate_temporal(self, matrices, num_timepoints=5):
        """Simulate temporal evolution from static matrices"""
        temporal_data = []
        
        for matrix in matrices:
            sequence = []
            for t in range(num_timepoints):
                # Add small perturbations to simulate time
                perturbed = matrix + np.random.randn(*matrix.shape) * 0.05
                perturbed = (perturbed + perturbed.T) / 2
                perturbed = np.clip(perturbed, -1, 1)
                sequence.append(perturbed)
            temporal_data.append(sequence)
        
        return np.array(temporal_data)
