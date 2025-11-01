"""
Brain atlas definitions (AAL - Automated Anatomical Labeling)
"""

import numpy as np


class AALAtlas:
    """
    AAL (Automated Anatomical Labeling) Brain Atlas
    90 cortical and subcortical regions
    """
    
    def __init__(self):
        self.num_regions = 90
        self.region_names = self._get_region_names()
        self.region_coordinates = self._get_region_coordinates()
        self.hemispheres = self._get_hemispheres()
    
    def _get_region_names(self):
        """Return AAL region names"""
        regions = [
            # Frontal (1-26)
            'Precentral_L', 'Precentral_R',
            'Frontal_Sup_L', 'Frontal_Sup_R',
            'Frontal_Sup_Orb_L', 'Frontal_Sup_Orb_R',
            'Frontal_Mid_L', 'Frontal_Mid_R',
            'Frontal_Mid_Orb_L', 'Frontal_Mid_Orb_R',
            'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R',
            'Frontal_Inf_Tri_L', 'Frontal_Inf_Tri_R',
            'Frontal_Inf_Orb_L', 'Frontal_Inf_Orb_R',
            'Rolandic_Oper_L', 'Rolandic_Oper_R',
            'Supp_Motor_Area_L', 'Supp_Motor_Area_R',
            'Olfactory_L', 'Olfactory_R',
            'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',
            'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R',
            
            # Parietal (27-48)
            'Paracentral_Lobule_L', 'Paracentral_Lobule_R',
            'Parietal_Sup_L', 'Parietal_Sup_R',
            'Parietal_Inf_L', 'Parietal_Inf_R',
            'SupraMarginal_L', 'SupraMarginal_R',
            'Angular_L', 'Angular_R',
            'Precuneus_L', 'Precuneus_R',
            'Postcentral_L', 'Postcentral_R',
            
            # Temporal (49-68)
            'Temporal_Sup_L', 'Temporal_Sup_R',
            'Temporal_Pole_Sup_L', 'Temporal_Pole_Sup_R',
            'Temporal_Mid_L', 'Temporal_Mid_R',
            'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R',
            'Temporal_Inf_L', 'Temporal_Inf_R',
            'Heschl_L', 'Heschl_R',
            
            # Occipital (69-80)
            'Calcarine_L', 'Calcarine_R',
            'Cuneus_L', 'Cuneus_R',
            'Lingual_L', 'Lingual_R',
            'Occipital_Sup_L', 'Occipital_Sup_R',
            'Occipital_Mid_L', 'Occipital_Mid_R',
            'Occipital_Inf_L', 'Occipital_Inf_R',
            'Fusiform_L', 'Fusiform_R',
            
            # Subcortical (81-90)
            'Hippocampus_L', 'Hippocampus_R',
            'Amygdala_L', 'Amygdala_R',
            'Caudate_L', 'Caudate_R',
            'Putamen_L', 'Putamen_R',
            'Thalamus_L', 'Thalamus_R'
        ]
        return regions
    
    def _get_region_coordinates(self):
        """
        Return approximate MNI coordinates for AAL regions
        This is a simplified representation
        """
        # Generate approximate coordinates
        # In practice, use actual MNI coordinates from atlas
        coords = np.random.randn(self.num_regions, 3) * 30
        
        # Ensure left/right hemisphere split
        for i in range(0, self.num_regions, 2):
            coords[i, 0] = -abs(coords[i, 0])      # Left hemisphere (negative X)
            coords[i+1, 0] = abs(coords[i+1, 0])   # Right hemisphere (positive X)
        
        return coords
    
    def _get_hemispheres(self):
        """Return hemisphere labels (L/R)"""
        hemispheres = []
        for name in self.region_names:
            if '_L' in name:
                hemispheres.append('L')
            elif '_R' in name:
                hemispheres.append('R')
            else:
                hemispheres.append('M')  # Midline
        return hemispheres
    
    def get_region_index(self, region_name):
        """Get index for region name"""
        try:
            return self.region_names.index(region_name)
        except ValueError:
            return None
    
    def get_hemisphere_regions(self, hemisphere='L'):
        """Get indices of regions in specified hemisphere"""
        indices = [i for i, h in enumerate(self.hemispheres) if h == hemisphere]
        return indices


def get_region_names():
    """Get AAL region names"""
    atlas = AALAtlas()
    return atlas.region_names


def get_region_coordinates():
    """Get AAL region coordinates"""
    atlas = AALAtlas()
    return atlas.region_coordinates
