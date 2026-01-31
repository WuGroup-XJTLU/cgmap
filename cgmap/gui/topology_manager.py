"""CG topology management for coarse-grained representations."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CGBond:
    """Represents a bond between two CG beads."""
    bead_i: int
    bead_j: int
    bond_type: int = 1


class TopologyManager:
    """Manage CG topology (bonds, angles, dihedrals) per molecule type."""

    def __init__(self):
        # mol_type_name -> list of CGBond
        self.bonds: dict[str, list[CGBond]] = {}

    def add_bond(self, mol_type: str, i: int, j: int, bond_type: int = 1) -> Optional[CGBond]:
        """Add a bond between beads i and j for the given molecule type.

        Returns the new CGBond, or None if the bond already exists.
        """
        if mol_type not in self.bonds:
            self.bonds[mol_type] = []

        # Check for duplicate (order-independent)
        for bond in self.bonds[mol_type]:
            if (bond.bead_i == i and bond.bead_j == j) or (bond.bead_i == j and bond.bead_j == i):
                return None

        new_bond = CGBond(bead_i=i, bead_j=j, bond_type=bond_type)
        self.bonds[mol_type].append(new_bond)
        return new_bond

    def remove_bond(self, mol_type: str, i: int, j: int) -> bool:
        """Remove the bond between beads i and j.

        Returns True if a bond was removed, False otherwise.
        """
        if mol_type not in self.bonds:
            return False

        original_len = len(self.bonds[mol_type])
        self.bonds[mol_type] = [
            bond for bond in self.bonds[mol_type]
            if not ((bond.bead_i == i and bond.bead_j == j) or (bond.bead_i == j and bond.bead_j == i))
        ]
        return len(self.bonds[mol_type]) < original_len

    def get_bonds(self, mol_type: str) -> list[CGBond]:
        """Get all bonds for the given molecule type."""
        return self.bonds.get(mol_type, [])

    def clear_bonds(self, mol_type: str):
        """Clear all bonds for the given molecule type."""
        self.bonds[mol_type] = []

    def validate_bonds(self, mol_type: str, n_beads: int):
        """Remove bonds with invalid bead indices.

        Called when beads are added/removed to clean up topology.
        """
        if mol_type not in self.bonds:
            return

        self.bonds[mol_type] = [
            bond for bond in self.bonds[mol_type]
            if 0 <= bond.bead_i < n_beads and 0 <= bond.bead_j < n_beads
        ]

    def infer_bonds_from_atomistic(self, mol_type: str, beads, local_bonds: list[tuple[int, int]]) -> int:
        """Infer CG bonds from atomistic connectivity.

        For each atomistic bond, find which CG beads contain each atom.
        If they belong to different beads, add a CG bond between those beads.

        Args:
            mol_type: Molecule type name.
            beads: List of BeadDefinition objects.
            local_bonds: List of (atom_i, atom_j) 0-indexed atomistic bonds.

        Returns:
            Number of bonds created.
        """
        self.clear_bonds(mol_type)

        # Build reverse map: atom_index -> bead_index
        atom_to_bead = {}
        for bead_idx, bead in enumerate(beads):
            for atom_idx in bead.atom_indices:
                atom_to_bead[atom_idx] = bead_idx

        bond_type_map = {}
        count = 0
        for atom_i, atom_j in local_bonds:
            bead_a = atom_to_bead.get(atom_i)
            bead_b = atom_to_bead.get(atom_j)
            if bead_a is not None and bead_b is not None and bead_a != bead_b:
                type_a = beads[bead_a].bead_type
                type_b = beads[bead_b].bead_type
                canon_key = tuple(sorted([type_a, type_b]))
                if canon_key not in bond_type_map:
                    bond_type_map[canon_key] = len(bond_type_map) + 1
                bond_type = bond_type_map[canon_key]
                result = self.add_bond(mol_type, bead_a, bead_b, bond_type=bond_type)
                if result is not None:
                    count += 1

        return count

    def infer_angles(self, mol_type: str) -> list[tuple[int, int, int]]:
        """Infer angles from bond adjacency.

        Returns list of (i, j, k) triplets where i-j and j-k are bonds.
        """
        bonds = self.get_bonds(mol_type)
        if not bonds:
            return []

        # Build adjacency list
        adj = {}
        for bond in bonds:
            i, j = bond.bead_i, bond.bead_j
            if i not in adj:
                adj[i] = []
            if j not in adj:
                adj[j] = []
            adj[i].append(j)
            adj[j].append(i)

        # Find all angles (i-j-k where i and k are neighbors of j, i < k)
        angles = []
        for j, neighbors in adj.items():
            for i in neighbors:
                for k in neighbors:
                    if i < k:  # Avoid duplicates
                        angles.append((i, j, k))

        return sorted(angles)

    def infer_dihedrals(self, mol_type: str) -> list[tuple[int, int, int, int]]:
        """Infer dihedrals from bond adjacency.

        Returns list of (i, j, k, l) quartets where i-j, j-k, k-l are bonds
        and i-j-k forms a proper angle.
        """
        bonds = self.get_bonds(mol_type)
        if not bonds:
            return []

        # Build adjacency list
        adj = {}
        for bond in bonds:
            i, j = bond.bead_i, bond.bead_j
            if i not in adj:
                adj[i] = []
            if j not in adj:
                adj[j] = []
            adj[i].append(j)
            adj[j].append(i)

        # Find all dihedrals (i-j-k-l where i-j, j-k, k-l are bonds)
        dihedrals = []
        for j, neighbors_j in adj.items():
            for k in neighbors_j:
                if k == j:
                    continue
                for i in neighbors_j:
                    if i == k:
                        continue
                    # Now i-j-k is an angle, find l connected to k
                    if k not in adj:
                        continue
                    for l in adj[k]:
                        if l == j:
                            continue
                        # Ensure unique ordering (avoid duplicates)
                        quartet = tuple(sorted([i, j, k, l]))
                        if quartet not in dihedrals:
                            dihedrals.append(quartet)

        return sorted(dihedrals)

    def get_summary(self, mol_type: str) -> dict:
        """Get topology summary for the given molecule type."""
        bonds = self.get_bonds(mol_type)
        angles = self.infer_angles(mol_type)
        dihedrals = self.infer_dihedrals(mol_type)

        return {
            'bonds': len(bonds),
            'angles': len(angles),
            'dihedrals': len(dihedrals),
        }
