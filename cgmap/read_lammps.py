import numpy as np
import sys
import logging


def check_dump_format(filepath):
    """Quick check of LAMMPS dump file format (first frame only).

    Returns dict with:
      - valid: bool
      - columns: list of column names (if found)
      - num_atoms: int (if found)
      - has_coords: bool (x,y,z present)
      - has_forces: bool (fx,fy,fz present)
      - has_images: bool (ix,iy,iz present)
      - has_mol: bool (mol column present)
      - errors: list of error strings
      - warnings: list of warning strings
    """
    result = {
        'valid': False,
        'columns': [],
        'num_atoms': 0,
        'has_coords': False,
        'has_forces': False,
        'has_images': False,
        'has_mol': False,
        'errors': [],
        'warnings': [],
    }

    try:
        with open(filepath, 'r') as f:
            # --- ITEM: TIMESTEP ---
            line = f.readline().strip()
            if not line.startswith('ITEM: TIMESTEP'):
                result['errors'].append(
                    f"Expected 'ITEM: TIMESTEP' on first line, got: '{line}'"
                )
                return result
            f.readline()  # skip timestep value

            # --- ITEM: NUMBER OF ATOMS ---
            line = f.readline().strip()
            if not line.startswith('ITEM: NUMBER OF ATOMS'):
                result['errors'].append(
                    f"Expected 'ITEM: NUMBER OF ATOMS', got: '{line}'"
                )
                return result
            try:
                num_atoms = int(f.readline().strip())
            except ValueError as e:
                result['errors'].append(f"Could not parse atom count: {e}")
                return result
            result['num_atoms'] = num_atoms

            # --- ITEM: BOX BOUNDS ---
            line = f.readline().strip()
            if not line.startswith('ITEM: BOX BOUNDS'):
                result['errors'].append(
                    f"Expected 'ITEM: BOX BOUNDS', got: '{line}'"
                )
                return result
            for _ in range(3):
                f.readline()  # skip box bound lines

            # --- ITEM: ATOMS ---
            line = f.readline().strip()
            if not line.startswith('ITEM: ATOMS'):
                result['errors'].append(
                    f"Expected 'ITEM: ATOMS', got: '{line}'"
                )
                return result

            columns = line.split()[2:]
            result['columns'] = columns

            # Check required columns
            required = {'id', 'type', 'x', 'y', 'z'}
            missing = required - set(columns)
            if missing:
                result['errors'].append(
                    f"Missing required columns: {sorted(missing)}"
                )
                return result

            result['has_coords'] = all(c in columns for c in ('x', 'y', 'z'))
            result['has_forces'] = all(c in columns for c in ('fx', 'fy', 'fz'))
            result['has_images'] = all(c in columns for c in ('ix', 'iy', 'iz'))
            result['has_mol'] = 'mol' in columns

            if not result['has_forces']:
                result['warnings'].append("No force columns (fx fy fz)")
            if not result['has_images']:
                result['warnings'].append("No image flag columns (ix iy iz)")

            # Validate first frame atom data
            expected_cols = len(columns)
            row_count = 0
            for _ in range(num_atoms):
                atom_line = f.readline()
                if not atom_line:
                    break
                parts = atom_line.strip().split()
                if len(parts) != expected_cols:
                    result['errors'].append(
                        f"Row {row_count + 1}: expected {expected_cols} columns, got {len(parts)}"
                    )
                    return result
                row_count += 1

            if row_count != num_atoms:
                result['errors'].append(
                    f"Expected {num_atoms} atom rows, found {row_count}"
                )
                return result

            result['valid'] = True

    except FileNotFoundError:
        result['errors'].append(f"File not found: {filepath}")
    except Exception as e:
        result['errors'].append(f"Error reading file: {e}")

    return result


def read_dump_file(filepath):
    """
    Read a LAMMPS dump file and organize data by timesteps into dictionaries and numpy arrays.
    
    Args:
        filepath (str): Path to the .dump file
        
    Returns:
        dict: Dictionary containing timestep data with numpy arrays
    """
    timesteps = {}
    current_timestep = None
    num_atoms = 0
    box_bounds = None
    header_items = None
    
    try:
        with open(filepath, 'r') as file:
            while True:
                line = file.readline()
                if not line:
                    break
                    
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                    
                # Check for ITEM: TIMESTEP
                if line.startswith("ITEM: TIMESTEP"):
                    if current_timestep is not None:
                        # Convert atom data to numpy array
                        timesteps[current_timestep]["atoms"] = np.array(atom_data)
                    
                    current_timestep = int(file.readline().strip())
                    timesteps[current_timestep] = {}
                    atom_data = []
                    
                # Get number of atoms
                elif line.startswith("ITEM: NUMBER OF ATOMS"):
                    num_atoms = int(file.readline().strip())
                    timesteps[current_timestep]["num_atoms"] = num_atoms
                    
                # Get box bounds
                elif line.startswith("ITEM: BOX BOUNDS"):
                    box_bounds = []
                    for _ in range(3):  # Assuming 3D system
                        bounds = file.readline().strip().split()
                        box_bounds.append([float(bounds[0]), float(bounds[1])])
                    timesteps[current_timestep]["box_bounds"] = np.array(box_bounds)
                    
                # Get atom data
                elif line.startswith("ITEM: ATOMS"):
                    # Parse header to know what columns represent
                    header_items = line.split()[2:]
                    timesteps[current_timestep]["columns"] = header_items
                    
                    # Read atom data
                    for _ in range(num_atoms):
                        atom_line = file.readline().strip().split()
                        atom_data.append([float(x) for x in atom_line])
                        
        # Convert last timestep's atom data
        if current_timestep is not None and atom_data:
            timesteps[current_timestep]["atoms"] = np.array(atom_data)
                    
        return timesteps
    
    except FileNotFoundError:
        logging.getLogger('cgmap').error(f"File '{filepath}' not found")
        return None
    except Exception as e:
        logging.getLogger('cgmap').error(f"Error reading dump file: {e}")
        return None

# Example usage
if __name__ == "__main__":
    dump_data = read_dump_file("20nsNVT_test.dump")
    if dump_data:
        # Print available timesteps
        print("Available timesteps:", list(dump_data.keys()))
        
        # Access data for first timestep
        first_timestep = min(dump_data.keys())
        print("\nFirst timestep data:")
        print("Number of atoms:", dump_data[first_timestep]["num_atoms"])
        print("Box bounds:\n", dump_data[first_timestep]["box_bounds"])
        print("Column headers:", dump_data[first_timestep]["columns"])
        print("Atom data shape:", dump_data[first_timestep]["atoms"].shape)

def get(dump_data,timestep,property):
    if dump_data and timestep in dump_data:
        if property=='xyz':
            # Get specific atom properties based on column headers
            columns = dump_data[timestep]["columns"]
            id_col_x = columns.index("x")
            id_col_y = columns.index("y")
            id_col_z = columns.index("z")
            # Get atom positions (assuming x, y, z are columns 3,4,5)
            xyz = dump_data[timestep]["atoms"][:,[id_col_x,id_col_y,id_col_z]]
            return xyz
        if property=='id':
            columns = dump_data[timestep]["columns"]
            id_col = columns.index("id")
            atom_ids = dump_data[timestep]["atoms"][:, id_col]
            return atom_ids
        if property=='type':
            columns = dump_data[timestep]["columns"]
            id_col = columns.index("type")
            atom_ids = dump_data[timestep]["atoms"][:, id_col]
            return atom_ids
        if property=='force':
            columns = dump_data[timestep]["columns"]
            id_col_x = columns.index("fx")
            id_col_y = columns.index("fy")
            id_col_z = columns.index("fz")
            # Get atom positions (assuming x, y, z are columns 3,4,5)
            xyz = dump_data[timestep]["atoms"][:,[id_col_x,id_col_y,id_col_z]]
            return xyz
        if property=='ixiyiz':
            columns = dump_data[timestep]["columns"]
            id_col_x = columns.index("ix")
            id_col_y = columns.index("iy")
            id_col_z = columns.index("iz")
            # Get atom positions (assuming x, y, z are columns 3,4,5)
            xyz = dump_data[timestep]["atoms"][:,[id_col_x,id_col_y,id_col_z]]
            return xyz
        raise ValueError(f"Unknown property: '{property}'. Supported properties: 'xyz', 'id', 'type', 'force', 'ixiyiz'")
    else:
        raise ValueError(f"Cannot access dump data for timestep {timestep}")

def get_frame(dump_data,property):
    if property=='xyz':
        # Get specific atom properties based on column headers
        columns = dump_data["columns"]
        id_col_x = columns.index("x")
        id_col_y = columns.index("y")
        id_col_z = columns.index("z")
        # Get atom positions (assuming x, y, z are columns 3,4,5)
        xyz = dump_data["atoms"][:,[id_col_x,id_col_y,id_col_z]]
        return xyz
    if property=='id':
        columns = dump_data["columns"]
        id_col = columns.index("id")
        atom_ids = dump_data["atoms"][:, id_col]
        return atom_ids
    if property=='box':
        bounds=dump_data["box_bounds"]
        box = np.array([bounds[0][1]-bounds[0][0],bounds[1][1]-bounds[1][0],bounds[2][1]-bounds[2][0]])
        return box
    if property=='type':
        columns = dump_data["columns"]
        id_col = columns.index("type")
        atom_ids = dump_data["atoms"][:, id_col]
        return atom_ids
    if property=='force':
        columns = dump_data["columns"]
        id_col_x = columns.index("fx")
        id_col_y = columns.index("fy")
        id_col_z = columns.index("fz")
        # Get atom positions (assuming x, y, z are columns 3,4,5)
        xyz = dump_data["atoms"][:,[id_col_x,id_col_y,id_col_z]]
        return xyz
    if property=='ixiyiz':
        columns = dump_data["columns"]
        id_col_x = columns.index("ix")
        id_col_y = columns.index("iy")
        id_col_z = columns.index("iz")
        # Get atom positions (assuming x, y, z are columns 3,4,5)
        xyz = dump_data["atoms"][:,[id_col_x,id_col_y,id_col_z]]
        return xyz
    raise ValueError(f"Unknown property: '{property}'. Supported properties: 'xyz', 'id', 'box', 'type', 'force', 'ixiyiz'")
