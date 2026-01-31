import os
import yaml


def export_mapping(mol_type, beads, output_dir, filename=None,
                   topology_manager=None, mol_type_name=None):
    """Export a single molecule type's mapping to YAML.

    Args:
        mol_type: MoleculeType instance.
        beads: List of BeadDefinition for this molecule type.
        output_dir: Directory to write the file.
        filename: Optional filename override (default: mapping_{mol_type.name}.yaml).
        topology_manager: Optional TopologyManager instance for CG topology.
        mol_type_name: Molecule type name for topology lookup.

    Returns:
        Path to the written file.
    """
    if not filename:
        filename = f"mapping_{mol_type.name}.yaml"

    site_types = {}
    sites_list = []

    for bead in beads:
        site_types[bead.name] = {
            'type': bead.bead_type,
            'index': [int(i) for i in bead.atom_indices],
            'x-weight': [float(w) for w in bead.x_weights],
            'f-weight': [float(w) for w in bead.f_weights],
        }
        # anchor is the first atom index of this bead within the molecule
        anchor = int(bead.atom_indices[0])
        sites_list.append([bead.name, anchor])

    data = {
        'site-types': site_types,
        'config': [{
            'anchor': 0,
            'repeat': mol_type.count,
            'offset': mol_type.atoms_per_mol,
            'sites': sites_list,
        }],
    }

    # Add topology section if available
    if topology_manager and mol_type_name:
        bonds = topology_manager.get_bonds(mol_type_name)
        if bonds:
            # Convert bonds from bead indices to bead names
            bond_list = []
            for bond in bonds:
                if 0 <= bond.bead_i < len(beads) and 0 <= bond.bead_j < len(beads):
                    name_i = beads[bond.bead_i].name
                    name_j = beads[bond.bead_j].name
                    bond_list.append([name_i, name_j, bond.bond_type])

            # Infer angles and dihedrals with type assignment
            bead_types = [b.bead_type for b in beads]

            angle_triplets = topology_manager.infer_angles(mol_type_name)
            angle_type_map = {}
            angle_list = []
            for (i, j, k) in angle_triplets:
                if 0 <= i < len(beads) and 0 <= j < len(beads) and 0 <= k < len(beads):
                    key = (bead_types[i], bead_types[j], bead_types[k])
                    canon_key = min(key, key[::-1])
                    if canon_key not in angle_type_map:
                        angle_type_map[canon_key] = len(angle_type_map) + 1
                    angle_list.append([beads[i].name, beads[j].name, beads[k].name, angle_type_map[canon_key]])

            dihedral_quartets = topology_manager.infer_dihedrals(mol_type_name)
            dihedral_type_map = {}
            dihedral_list = []
            for (i, j, k, l) in dihedral_quartets:
                if 0 <= i < len(beads) and 0 <= j < len(beads) and 0 <= k < len(beads) and 0 <= l < len(beads):
                    key = (bead_types[i], bead_types[j], bead_types[k], bead_types[l])
                    canon_key = min(key, key[::-1])
                    if canon_key not in dihedral_type_map:
                        dihedral_type_map[canon_key] = len(dihedral_type_map) + 1
                    dihedral_list.append([beads[i].name, beads[j].name, beads[k].name, beads[l].name, dihedral_type_map[canon_key]])

            data['topology'] = {
                'bonds': bond_list,
                'angles': angle_list,
                'dihedrals': dihedral_list,
            }

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False)

    return filepath


def export_system(mol_types, beads_by_mol, output_dir, filename='system.yaml',
                  topology_manager=None):
    """Export the system YAML that references all mapping files.

    Args:
        mol_types: List of MoleculeType instances.
        beads_by_mol: Dict mapping mol_type.name -> list of BeadDefinition.
        output_dir: Directory to write files.
        filename: System YAML filename.
        topology_manager: Optional TopologyManager instance for CG topology.

    Returns:
        Path to the system YAML file.
    """
    names = []
    numbers = []

    for mol_type in mol_types:
        if mol_type.name not in beads_by_mol or not beads_by_mol[mol_type.name]:
            continue
        mapping_file = f"mapping_{mol_type.name}.yaml"
        export_mapping(mol_type, beads_by_mol[mol_type.name], output_dir, mapping_file,
                      topology_manager=topology_manager, mol_type_name=mol_type.name)
        names.append(mapping_file)
        numbers.append(1)

    system_data = {
        'system': {
            'names': names,
            'numbers': numbers,
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        yaml.dump(system_data, f, default_flow_style=None, sort_keys=False)

    return filepath
