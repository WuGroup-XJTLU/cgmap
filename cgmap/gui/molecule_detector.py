import numpy as np
from dataclasses import dataclass
from cgmap.read_lammps import read_dump_file, get_frame


@dataclass
class MoleculeType:
    name: str
    signature: tuple
    count: int
    atoms_per_mol: int
    start_index: int
    representative_xyz: np.ndarray
    representative_types: np.ndarray


class MoleculeDetector:
    """Detect unique molecule types from a LAMMPS dump frame."""

    def __init__(self, frame_data):
        self.frame_data = frame_data
        self.types = get_frame(frame_data, 'type').astype(int)
        self.xyz = get_frame(frame_data, 'xyz')
        self.columns = frame_data['columns']
        self.num_atoms = len(self.types)

    def detect(self, atoms_per_mol_hint=None):
        """Detect molecule types.

        Args:
            atoms_per_mol_hint: Optional string like "12,3" giving atoms per
                molecule for each molecule type in order.

        Returns:
            List of MoleculeType instances.
        """
        if 'mol' in self.columns:
            return self._detect_by_mol_id()

        if atoms_per_mol_hint:
            sizes = [int(s.strip()) for s in atoms_per_mol_hint.split(',')]
            return self._detect_by_hint(sizes)

        return self._detect_by_pattern()

    def _detect_by_mol_id(self):
        mol_col = self.columns.index('mol')
        mol_ids = self.frame_data['atoms'][:, mol_col].astype(int)

        unique_mols = np.unique(mol_ids)
        sig_map = {}

        for mol_id in unique_mols:
            mask = mol_ids == mol_id
            mol_types = tuple(self.types[mask])
            mol_xyz = self.xyz[mask]
            start = int(np.where(mask)[0][0])

            if mol_types not in sig_map:
                sig_map[mol_types] = {
                    'count': 0,
                    'start_index': start,
                    'xyz': mol_xyz,
                    'types': self.types[mask],
                }
            sig_map[mol_types]['count'] += 1

        return self._build_result(sig_map)

    def _detect_by_pattern(self):
        """Find repeating subsequences in the type array."""
        types_list = list(self.types)
        sig_map = {}
        offset = 0

        while offset < len(types_list):
            remaining = types_list[offset:]
            block_size = self._find_repeat_length(remaining)
            block = tuple(remaining[:block_size])

            count = 0
            pos = offset
            while pos + block_size <= len(types_list):
                candidate = tuple(types_list[pos:pos + block_size])
                if candidate != block:
                    break
                count += 1
                pos += block_size

            if block not in sig_map:
                sig_map[block] = {
                    'count': 0,
                    'start_index': offset,
                    'xyz': self.xyz[offset:offset + block_size],
                    'types': self.types[offset:offset + block_size],
                }
            sig_map[block]['count'] += count
            offset = pos

        return self._build_result(sig_map)

    def _find_repeat_length(self, seq):
        """Find the shortest repeating block length that uses multiple types.

        For LAMMPS dumps, molecules typically contain multiple atom types
        (e.g. C and H in benzene). A single-type repeating unit (e.g. just
        type 1) is almost never a real molecule, so we prefer the shortest
        block that contains more than one unique type and repeats at least
        twice. If no multi-type block repeats, fall back to shortest
        single-type repeating block.
        """
        n = len(seq)
        best_single = None

        for size in range(1, n + 1):
            block = seq[:size]
            reps = 0
            for i in range(0, n, size):
                if seq[i:i + size] == block:
                    reps += 1
                else:
                    break
            if reps > 1:
                if len(set(block)) > 1:
                    return size
                if best_single is None:
                    best_single = size

        return best_single if best_single is not None else n

    def _detect_by_hint(self, sizes):
        """Use user-provided atoms-per-molecule sizes."""
        sig_map = {}
        offset = 0

        for size in sizes:
            if offset >= self.num_atoms:
                break

            block = tuple(self.types[offset:offset + size])

            # Count consecutive repeats
            count = 0
            pos = offset
            while pos + size <= self.num_atoms:
                candidate = tuple(self.types[pos:pos + size])
                if candidate != block:
                    break
                count += 1
                pos += size

            sig_map[block] = {
                'count': count,
                'start_index': offset,
                'xyz': self.xyz[offset:offset + size],
                'types': self.types[offset:offset + size],
            }
            offset = pos

        return self._build_result(sig_map)

    def _build_result(self, sig_map):
        result = []
        for idx, (sig, info) in enumerate(sig_map.items(), 1):
            result.append(MoleculeType(
                name=f"MolType_{idx}",
                signature=sig,
                count=info['count'],
                atoms_per_mol=len(sig),
                start_index=info['start_index'],
                representative_xyz=info['xyz'],
                representative_types=info['types'],
            ))
        return result
