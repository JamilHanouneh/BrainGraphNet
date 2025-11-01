"""
Synthetic Brain Connectivity Data Generator
Generates realistic temporal connectivity matrices with evolution
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm


class SyntheticBrainDataGenerator:
    """Generate synthetic brain connectivity data with temporal evolution"""
    
    def __init__(self, config):
        self.config = config
        self.synth_config = config['data']['synthetic']
        self.num_subjects = self.synth_config['num_subjects']
        self.num_timepoints = self.synth_config['num_timepoints']
        self.num_regions = self.synth_config['num_regions']
        self.noise_level = self.synth_config['noise_level']
        self.disease_progression = self.synth_config['disease_progression']
        self.save_path = Path(self.synth_config['save_path'])
        
        # Set seed
        np.random.seed(config['system']['seed'])
    
    def generate(self):
        """Generate complete synthetic dataset"""
        print(f"Generating synthetic data for {self.num_subjects} subjects...")
        
        all_connectivity = []
        all_labels = []
        
        for subject_id in tqdm(range(self.num_subjects), desc="Generating subjects"):
            # Determine if subject has disease
            has_disease = subject_id >= (self.num_subjects // 2)
            
            # Generate temporal connectivity
            connectivity_sequence = self._generate_subject_sequence(has_disease)
            all_connectivity.append(connectivity_sequence)
            all_labels.append(1 if has_disease else 0)
        
        connectivity_data = np.array(all_connectivity)
        labels = np.array(all_labels)
        
        # Save
        self.save_path.mkdir(parents=True, exist_ok=True)
        np.save(self.save_path / 'connectivity_data.npy', connectivity_data)
        np.save(self.save_path / 'labels.npy', labels)
        
        print(f"✅ Data saved to {self.save_path}")
        return connectivity_data, labels
    
    def _generate_subject_sequence(self, has_disease=False):
        """Generate temporal connectivity sequence for one subject"""
        # Base connectivity structure
        base_connectivity = self._generate_base_connectivity()
        
        sequence = []
        for t in range(self.num_timepoints):
            # Apply temporal evolution
            connectivity_t = base_connectivity.copy()
            
            # Add age/time effect
            connectivity_t = self._apply_time_effect(connectivity_t, t)
            
            # Add disease progression if applicable
            if has_disease and self.disease_progression:
                connectivity_t = self._apply_disease_effect(connectivity_t, t)
            
            # Add noise
            connectivity_t = self._add_noise(connectivity_t)
            
            # Ensure symmetry and valid range
            connectivity_t = self._postprocess_matrix(connectivity_t)
            
            sequence.append(connectivity_t)
        
        return np.array(sequence)
    
    def _generate_base_connectivity(self):
        """Generate base connectivity matrix with realistic structure"""
        # Start with random correlations
        connectivity = np.random.randn(self.num_regions, self.num_regions)
        connectivity = (connectivity + connectivity.T) / 2  # Make symmetric
        
        # Add modular structure (brain regions cluster)
        num_modules = 6
        module_size = self.num_regions // num_modules
        
        for i in range(num_modules):
            start = i * module_size
            end = start + module_size
            # Increase within-module connectivity
            connectivity[start:end, start:end] += 0.5
        
        # Normalize to correlation range
        connectivity = np.tanh(connectivity * 0.5)
        
        # Set diagonal to 1 (self-correlation)
        np.fill_diagonal(connectivity, 1.0)
        
        return connectivity
    
    def _apply_time_effect(self, connectivity, timepoint):
        """Apply temporal evolution (aging, development)"""
        # Gradual decrease in connectivity strength
        decay_factor = 1.0 - (timepoint * 0.02)
        connectivity *= decay_factor
        
        # Add small random walk
        drift = np.random.randn(self.num_regions, self.num_regions) * 0.01
        drift = (drift + drift.T) / 2
        connectivity += drift
        
        return connectivity
    
    def _apply_disease_effect(self, connectivity, timepoint):
        """Apply disease progression effect"""
        # Disease progresses over time
        disease_strength = timepoint / self.num_timepoints
        
        # Selectively weaken certain connections (disease-affected regions)
        affected_regions = np.random.choice(
            self.num_regions, 
            size=self.num_regions // 5, 
            replace=False
        )
        
        for region in affected_regions:
            connectivity[region, :] *= (1.0 - disease_strength * 0.3)
            connectivity[:, region] *= (1.0 - disease_strength * 0.3)
        
        return connectivity
    
    def _add_noise(self, connectivity):
        """Add measurement noise"""
        noise = np.random.randn(self.num_regions, self.num_regions) * self.noise_level
        noise = (noise + noise.T) / 2  # Symmetric noise
        return connectivity + noise
    
    def _postprocess_matrix(self, connectivity):
        """Ensure matrix is valid connectivity matrix"""
        # Make symmetric
        connectivity = (connectivity + connectivity.T) / 2
        
        # Clip to valid correlation range
        connectivity = np.clip(connectivity, -1, 1)
        
        # Set diagonal to 1
        np.fill_diagonal(connectivity, 1.0)
        
        return connectivity
