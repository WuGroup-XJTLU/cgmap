import numpy as np
import logging
from collections import deque

logger = logging.getLogger('cgmap')


def unwrap_via_image_flags(xyz, ixiyiz, box):
    """Unwrap coordinates using image flags: xyz + ixiyiz * box."""
    return xyz + ixiyiz * box


def parse_lammps_data_bonds(filepath):
    """Read the Bonds section from a LAMMPS data file.

    Returns list of (atom_i, atom_j) tuples with 1-indexed atom IDs.
    """
    bonds = []
    in_bonds = False
    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped == 'Bonds':
                # skip the blank line after section header
                next(f)
                in_bonds = True
                continue
            if in_bonds:
                if stripped == '' or stripped.split()[0].isalpha():
                    break
                parts = stripped.split()
                # format: bond-ID bond-type atom1 atom2
                atom_i = int(parts[2])
                atom_j = int(parts[3])
                bonds.append((atom_i, atom_j))
    return bonds


def build_molecule_graphs(bonds, n_atoms):
    """BFS to find connected components from bond list.

    Args:
        bonds: list of (atom_i, atom_j) tuples, 1-indexed
        n_atoms: total number of atoms

    Returns:
        molecules: list of lists of atom indices (0-indexed)
        adj: dict mapping atom index to set of bonded atom indices (0-indexed)
    """
    adj = {i: set() for i in range(n_atoms)}
    for ai, aj in bonds:
        adj[ai - 1].add(aj - 1)
        adj[aj - 1].add(ai - 1)

    visited = set()
    molecules = []
    for i in range(n_atoms):
        if i in visited:
            continue
        mol = []
        queue = deque([i])
        visited.add(i)
        while queue:
            atom = queue.popleft()
            mol.append(atom)
            for nb in adj[atom]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        molecules.append(mol)
    return molecules, adj


def unwrap_via_topology(xyz, box, molecules, adj):
    """Unwrap coordinates using bond topology and minimum image convention.

    BFS per molecule: each atom is unwrapped relative to its bonded parent.
    """
    unwrapped = xyz.copy()
    for mol in molecules:
        if len(mol) <= 1:
            continue
        root = mol[0]
        visited = {root}
        queue = deque([root])
        while queue:
            current = queue.popleft()
            for nb in adj[current]:
                if nb not in visited and nb in set(mol):
                    visited.add(nb)
                    queue.append(nb)
                    delta = unwrapped[nb] - unwrapped[current]
                    # minimum image correction
                    delta -= box * np.round(delta / box)
                    unwrapped[nb] = unwrapped[current] + delta
    return unwrapped


def unwrap_frame(frame_data, data_file=None, topology_cache=None):
    """Unwrap a single frame's coordinates.

    Strategy A: use image flags if available.
    Strategy B: use bond topology from LAMMPS data file.
    Falls back to raw coordinates with a warning.

    Args:
        frame_data: dict for one timestep from read_dump_file
        data_file: path to LAMMPS data file (for topology fallback)
        topology_cache: dict with 'molecules' and 'adj' keys, or None

    Returns:
        (unwrapped_xyz, topology_cache)
    """
    from cgmap.read_lammps import get_frame

    columns = frame_data['columns']
    xyz = get_frame(frame_data, 'xyz')
    box = get_frame(frame_data, 'box')

    # Strategy A: image flags
    has_images = all(c in columns for c in ('ix', 'iy', 'iz'))
    if has_images:
        ixiyiz = get_frame(frame_data, 'ixiyiz')
        unwrapped = unwrap_via_image_flags(xyz, ixiyiz, box)
        _replace_xyz(frame_data, unwrapped)
        return unwrapped, topology_cache

    # Strategy B: topology from data file
    if data_file is not None:
        if topology_cache is None:
            bonds = parse_lammps_data_bonds(data_file)
            n_atoms = frame_data['num_atoms']
            molecules, adj = build_molecule_graphs(bonds, n_atoms)
            topology_cache = {'molecules': molecules, 'adj': adj}

        # Build id-to-index mapping since dump atoms may not be sorted by ID
        atom_ids = get_frame(frame_data, 'id')
        id_to_idx = {int(aid): idx for idx, aid in enumerate(atom_ids)}

        # Remap topology to current frame ordering
        molecules_mapped = []
        for mol in topology_cache['molecules']:
            molecules_mapped.append([id_to_idx[atom_0idx + 1] for atom_0idx in mol])

        adj_mapped = {}
        for atom, neighbors in topology_cache['adj'].items():
            mapped_atom = id_to_idx[atom + 1]
            adj_mapped[mapped_atom] = {id_to_idx[nb + 1] for nb in neighbors}

        unwrapped = unwrap_via_topology(xyz, box, molecules_mapped, adj_mapped)
        _replace_xyz(frame_data, unwrapped)
        return unwrapped, topology_cache

    # No unwrapping possible
    logger.warning(
        "No image flags in dump and no --data file provided. "
        "Coordinates are NOT unwrapped; CG mapping may be incorrect for "
        "molecules that cross periodic boundaries."
    )
    return xyz, topology_cache


def _replace_xyz(frame_data, new_xyz):
    """Replace x, y, z columns in frame_data['atoms'] in-place."""
    columns = frame_data['columns']
    ix = columns.index('x')
    iy = columns.index('y')
    iz = columns.index('z')
    frame_data['atoms'][:, ix] = new_xyz[:, 0]
    frame_data['atoms'][:, iy] = new_xyz[:, 1]
    frame_data['atoms'][:, iz] = new_xyz[:, 2]
