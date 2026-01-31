import copy
from dataclasses import dataclass, field


COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000',
    '#000075', '#a9a9a9',
]


@dataclass
class BeadDefinition:
    name: str
    bead_type: str
    atom_indices: list
    x_weights: list
    f_weights: list
    color: str
    is_propagated: bool = False


class BeadManager:
    """Track bead definitions per molecule type."""

    def __init__(self):
        # mol_type_name -> list of BeadDefinition
        self.beads: dict[str, list[BeadDefinition]] = {}
        self._color_idx: dict[str, int] = {}

    def add_bead(self, mol_type_name, bead_name, atom_indices, x_weights,
                 f_weights=None, is_propagated=False, bead_type=None):
        if mol_type_name not in self.beads:
            self.beads[mol_type_name] = []
            self._color_idx[mol_type_name] = 0

        if f_weights is None:
            f_weights = [1.0] * len(atom_indices)

        color = COLORS[self._color_idx[mol_type_name] % len(COLORS)]
        self._color_idx[mol_type_name] += 1

        bead = BeadDefinition(
            name=bead_name,
            bead_type=bead_type or bead_name,
            atom_indices=list(atom_indices),
            x_weights=list(x_weights),
            f_weights=list(f_weights),
            color=color,
            is_propagated=is_propagated,
        )
        self.beads[mol_type_name].append(bead)
        return bead

    def remove_bead(self, mol_type_name, bead_name):
        if mol_type_name in self.beads:
            self.beads[mol_type_name] = [
                b for b in self.beads[mol_type_name] if b.name != bead_name
            ]

    def clear_beads(self, mol_type_name):
        self.beads[mol_type_name] = []
        self._color_idx[mol_type_name] = 0

    def get_beads(self, mol_type_name):
        return self.beads.get(mol_type_name, [])

    def get_assigned_indices(self, mol_type_name):
        assigned = set()
        for bead in self.get_beads(mol_type_name):
            assigned.update(bead.atom_indices)
        return assigned

    def get_unassigned_indices(self, mol_type_name, total_atoms):
        assigned = self.get_assigned_indices(mol_type_name)
        return sorted(set(range(total_atoms)) - assigned)

    def snapshot(self, mol_type_name):
        """Return a deep copy of current beads and color index for a molecule type."""
        beads = copy.deepcopy(self.beads.get(mol_type_name, []))
        color_idx = self._color_idx.get(mol_type_name, 0)
        return (beads, color_idx)

    def restore(self, mol_type_name, snapshot):
        """Restore beads and color index from a snapshot."""
        beads, color_idx = snapshot
        self.beads[mol_type_name] = copy.deepcopy(beads)
        self._color_idx[mol_type_name] = color_idx
