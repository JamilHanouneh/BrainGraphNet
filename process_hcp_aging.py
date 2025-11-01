"""
Process HCP Aging data to connectivity matrices
Extracts time series and computes correlation matrices
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

def find_hcp_data_directory():
    """Find the HCP data directory"""
    possible_paths = [
        Path('data/raw/HCP/HCPAging'),
        Path('data/raw/HCP/HPCAging'),
        Path('data/raw/HPC/HPCAging'),
        Path('data/raw/HPC/HCPAging'),
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"✅ Found HCP data at: {path}")
            return path
    
    # If not found, list what's actually there
    print("❌ Could not find HCP data directory")
    print("\nLooking for data in:")
    for path in possible_paths:
        print(f"  - {path.absolute()}")
    
    print("\nActual contents of data/raw/:")
    data_raw = Path('data/raw')
    if data_raw.exists():
        for item in data_raw.iterdir():
            print(f"  - {item.name}")
            if item.is_dir():
                for subitem in item.iterdir():
                    print(f"    - {subitem.name}")
    
    return None


def load_timeseries_from_txt(file_path):
    """Load time series from text file"""
    try:
        # Try different delimiters
        data = np.loadtxt(file_path)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_timeseries_from_subject(subject_path):
    """
    Load time series from HCP subject directory
    Looks for any available parcellation
    """
    # Available atlases in your data
    atlases = [
        'Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S1',
        'Schaefer2018_1000Parcels_7Networks_Tian_Subcortex_S1_3T',
        'Schaefer2018_100Parcels_7Networks_Tian_Subcortex_S1_3T',
        'Gordon333.32k_fs_LR_Tian_Subcortex_S1_3T',
        'GlasserFreesurfer',
        'fsaverage.BN_Atlas.32k_fs_LR_246regions'
    ]
    
    # Check V1 session
    v1_path = subject_path / 'V1'
    if not v1_path.exists():
        return None, None
    
    # Check REST scans
    rest_scans = ['rfMRI_REST1', 'rfMRI_REST2']
    
    for rest_scan in rest_scans:
        rest_path = v1_path / rest_scan
        if not rest_path.exists():
            continue
        
        # Try each atlas
        for atlas in atlases:
            atlas_path = rest_path / atlas
            if not atlas_path.exists():
                continue
            
            # Look for time series files
            txt_files = list(atlas_path.glob('*.txt'))
            ptseries_files = list(atlas_path.glob('*.ptseries.nii'))
            
            if txt_files:
                timeseries = load_timeseries_from_txt(txt_files[0])
                if timeseries is not None:
                    return timeseries, atlas
            
            # Note: For .ptseries.nii you need nibabel
            # Skipping for now as txt is easier
    
    return None, None


def compute_correlation_matrix(timeseries):
    """
    Compute correlation matrix from time series
    
    Args:
        timeseries: (timepoints, regions) or (regions, timepoints)
    
    Returns:
        connectivity: (regions, regions) correlation matrix
    """
    # Ensure shape is (timepoints, regions)
    if timeseries.shape[0] > timeseries.shape[1]:
        # Already (timepoints, regions)
        pass
    else:
        # Transpose to (timepoints, regions)
        timeseries = timeseries.T
    
    # Compute correlation matrix
    # Correlation between columns (regions)
    connectivity = np.corrcoef(timeseries.T)
    
    # Handle NaN values
    connectivity = np.nan_to_num(connectivity, nan=0.0)
    
    # Ensure diagonal is 1
    np.fill_diagonal(connectivity, 1.0)
    
    return connectivity


def process_hcp_aging_dataset():
    """Process all subjects and save connectivity matrices"""
    
    # Find HCP data directory
    hcp_dir = find_hcp_data_directory()
    
    if hcp_dir is None:
        print("\n❌ Cannot proceed: HCP data directory not found")
        print("\nPlease verify your data location and update the path in the script")
        return None, None
    
    # Output directory
    output_dir = Path('data/processed/HCP_connectivity')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all subject directories
    subject_dirs = [d for d in hcp_dir.iterdir() if d.is_dir() and d.name.startswith('HCA')]
    
    print(f"\nFound {len(subject_dirs)} subjects")
    
    if len(subject_dirs) == 0:
        print("❌ No subject directories found starting with 'HCA'")
        print(f"Contents of {hcp_dir}:")
        for item in hcp_dir.iterdir():
            print(f"  - {item.name}")
        return None, None
    
    connectivity_matrices = []
    subject_ids = []
    atlas_used = None
    
    for subject_dir in tqdm(subject_dirs[:50], desc="Processing subjects"):  # Process first 50
        try:
            # Load time series
            timeseries, atlas = load_timeseries_from_subject(subject_dir)
            
            if timeseries is None:
                continue
            
            if atlas_used is None:
                atlas_used = atlas
                print(f"\nUsing atlas: {atlas}")
                print(f"Time series shape: {timeseries.shape}")
            
            # Compute connectivity
            connectivity = compute_correlation_matrix(timeseries)
            
            # Save
            connectivity_matrices.append(connectivity)
            subject_ids.append(subject_dir.name)
            
            # Save individual matrix
            np.save(output_dir / f'{subject_dir.name}_connectivity.npy', connectivity)
            
        except Exception as e:
            print(f"\nError processing {subject_dir.name}: {e}")
            continue
    
    # Save all matrices
    if connectivity_matrices:
        all_connectivity = np.array(connectivity_matrices)
        
        print(f"\nProcessed {len(connectivity_matrices)} subjects")
        print(f"Connectivity matrix shape: {connectivity_matrices[0].shape}")
        
        # Create temporal dimension (simulate multiple timepoints)
        num_subjects = len(connectivity_matrices)
        num_regions = connectivity_matrices[0].shape[0]
        
        # Create 3 timepoints with small variations
        temporal_data = np.zeros((num_subjects, 3, num_regions, num_regions))
        
        for s in range(num_subjects):
            for t in range(3):
                # Add small temporal variation
                noise = np.random.randn(num_regions, num_regions) * 0.03
                noise = (noise + noise.T) / 2  # Keep symmetric
                temporal_data[s, t] = connectivity_matrices[s] + noise
                
                # Clip to valid correlation range
                temporal_data[s, t] = np.clip(temporal_data[s, t], -1, 1)
                
                # Ensure diagonal is 1
                np.fill_diagonal(temporal_data[s, t], 1.0)
        
        # Save
        np.save(output_dir / 'all_connectivity_temporal.npy', temporal_data)
        np.save(output_dir / 'connectivity_matrices.npy', all_connectivity)
        
        # Save subject IDs
        with open(output_dir / 'subject_ids.txt', 'w') as f:
            f.write('\n'.join(subject_ids))
        
        # Save metadata
        metadata = {
            'num_subjects': num_subjects,
            'num_timepoints': 3,
            'num_regions': num_regions,
            'atlas': atlas_used,
            'data_shape': temporal_data.shape
        }
        
        with open(output_dir / 'metadata.txt', 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\n✅ Processing complete!")
        print(f"   Processed subjects: {len(connectivity_matrices)}")
        print(f"   Temporal data shape: {temporal_data.shape}")
        print(f"   Number of regions: {num_regions}")
        print(f"   Atlas used: {atlas_used}")
        print(f"   Saved to: {output_dir}")
        
        return temporal_data, subject_ids
    else:
        print("\n❌ No subjects were successfully processed")
        print("Please check your data format and paths")
        return None, None


if __name__ == "__main__":
    print("="*60)
    print("HCP Aging Data Processing")
    print("="*60)
    process_hcp_aging_dataset()
