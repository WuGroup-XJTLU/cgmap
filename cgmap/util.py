import numpy as np
import logging

"""
# Example usage of the loaded NPZ data
loaded_data = load_cg_data_npz('cg_system.npz')
if loaded_data:
    # Access the data
    first_frame_coords = loaded_data['R'][0]
    first_frame_sites = loaded_data['z'][0]
    first_frame_forces = loaded_data['F'][0]
    first_frame_box = loaded_data['cell'][0]
"""

# Add a helper function to read the NPZ file (for future use):
def load_cg_data_npz(filename):
    """
    Load coarse-grained data from NPZ file format.
    
    Args:
        filename (str): Input NPZ filename
        
    Returns:
        dict: Dictionary containing CG data
    """
    try:
        data = np.load(filename, allow_pickle=True)
        cg_data = {
            'R': {i: coords for i, coords in enumerate(data['coordinates'])},
            'F': {i: forces for i, forces in enumerate(data['forces'])},
            'z': {i: sites for i, sites in enumerate(data['site_types'])},
            'cell': {i: box for i, box in enumerate(data['cell'])}
        }
        return cg_data
    except Exception as e:
        logging.getLogger('cgmap').error(f"Error loading NPZ file: {e}")
        return None