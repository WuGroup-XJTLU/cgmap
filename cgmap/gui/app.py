import os
import panel as pn
import numpy as np
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from cgmap.read_lammps import read_dump_file, check_dump_format, get_frame
from cgmap.unwrap import unwrap_frame, parse_lammps_data_bonds
from cgmap.gui.molecule_detector import MoleculeDetector
from cgmap.gui.viewer import MolecularViewer
from cgmap.gui.bead_manager import BeadManager
from cgmap.gui.yaml_exporter import export_system
from cgmap.gui.topology_manager import TopologyManager
from cgmap.gui.cg_viewer import CGViewer, compute_bead_positions
from cgmap.cgmap import (
    cg_map, wrap_coordinates, write_xyz_trajectory,
    save_cg_data_npz, write_lammps_data, read_mapping_file,
)

pn.extension(sizing_mode='stretch_width')

# ========== ELEGANT LIGHT THEME - Flowing Water Over Hills ==========
ELEGANT_LIGHT_CSS = """
/* ========== Foundation ========== */
:root {
    --bg-primary: #FAFBFC;
    --bg-secondary: #FFFFFF;
    --bg-tertiary: #F0F2F5;
    --text-primary: #1A1A2E;
    --text-secondary: #4A5568;
    --text-tertiary: #718096;
    --border-subtle: #E2E8F0;
    --border-light: #EDF2F7;
    --accent-primary: #5B8DB8;
    --accent-hover: #4A7BA4;
    --accent-light: #E8F0F7;
    --success: #6B9B8A;
    --success-light: #E6F0EC;
    --warning: #C9A66B;
    --warning-light: #F5F0E6;
    --danger: #B87B7B;
    --danger-light: #F5E8E8;
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.03);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.02);
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --transition-fast: 150ms ease;
    --transition-base: 250ms ease;
}

body, html { background-color: var(--bg-primary) !important; color: var(--text-primary) !important; font-family: 'Inter', -apple-system, sans-serif !important; -webkit-font-smoothing: antialiased; }
.bk-root, .bk-root * { font-family: 'Inter', -apple-system, sans-serif !important; }
.bk-root { color: var(--text-primary) !important; }
.bk-input-group > label { color: var(--text-secondary) !important; font-size: 0.6875rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; margin-bottom: 6px !important; }

/* Nature Style Buttons */
.bk-btn { background: linear-gradient(180deg, #F5F7F9 0%, #E8ECEF 100%) !important; color: #4A5568 !important; border: 1px solid #D1D9E0 !important; border-radius: var(--radius-md) !important; font-weight: 500 !important; font-size: 0.8125rem !important; padding: 8px 16px !important; box-shadow: 0 2px 4px rgba(0,0,0,0.04) !important; transition: all var(--transition-fast) !important; }
.bk-btn:hover { background: linear-gradient(180deg, #E8ECEF 0%, #DBE0E5 100%) !important; border-color: #C4CCD4 !important; box-shadow: 0 4px 8px rgba(0,0,0,0.06) !important; transform: translateY(-1px) !important; }
.bk-btn.bk-btn-primary { background: linear-gradient(180deg, #5B8DB8 0%, #4A7BA4 100%) !important; color: #FFFFFF !important; border-color: #4A7BA4 !important; box-shadow: 0 2px 4px rgba(74, 123, 164, 0.2) !important; }
.bk-btn.bk-btn-primary:hover { background: linear-gradient(180deg, #4A7BA4 0%, #3D6A8F 100%) !important; }
.bk-btn.bk-btn-success { background: linear-gradient(180deg, #6B9B8A 0%, #5A8A79 100%) !important; color: #FFFFFF !important; border-color: #5A8A79 !important; }
.bk-btn.bk-btn-warning { background: linear-gradient(180deg, #C9A66B 0%, #B8955A 100%) !important; color: #FFFFFF !important; }
.bk-btn.bk-btn-danger { background: linear-gradient(180deg, #B87B7B 0%, #A86A6A 100%) !important; color: #FFFFFF !important; }

.bk-input { background-color: var(--bg-secondary) !important; color: var(--text-primary) !important; border: 1px solid var(--border-subtle) !important; border-radius: var(--radius-md) !important; }
.bk-input:focus { border-color: var(--accent-primary) !important; box-shadow: 0 0 0 3px var(--accent-light) !important; }
.bk-select { background-color: var(--bg-secondary) !important; border: 1px solid var(--border-subtle) !important; border-radius: var(--radius-md) !important; }

.alert-info { background: linear-gradient(135deg, #E8F0F7 0%, #D5E3EE 100%) !important; border-color: #B8D4E8 !important; color: #3D5A73 !important; }
.alert-success { background: linear-gradient(135deg, #E6F0EC 0%, #D4E5DE 100%) !important; border-color: #B3D4C8 !important; color: #3D5A4F !important; }
.alert-warning { background: linear-gradient(135deg, #F5F0E6 0%, #EDE4D4 100%) !important; border-color: #DDD0B8 !important; color: #6B5A3D !important; }
.alert-danger { background: linear-gradient(135deg, #F5E8E8 0%, #EDD8D8 100%) !important; border-color: #DDB8B8 !important; color: #6B3D3D !important; }

.left-sidebar, .right-sidebar { background: var(--bg-secondary) !important; border-radius: var(--radius-lg) !important; box-shadow: var(--shadow-md) !important; border: 1px solid var(--border-light) !important; }
"""
pn.config.raw_css.append(ELEGANT_LIGHT_CSS)
pn.config.raw_css.append("@import url('https://fonts.googleapis.com/css2?family=Caveat:wght@600;700&display=swap');")

class CGMapApp:
    """Main GUI application for interactive CG mapping."""

    def __init__(self):
        self.dump_data = None
        self.mol_types = []
        self.current_mol_type = None
        self.bead_manager = BeadManager()
        self.atom_masses = {}  # atom_type_int -> mass
        self.all_bonds_global = None  # raw bonds from data file, list of (atom_i, atom_j) 1-indexed

        # Propagation state
        self._pending_propagation = []  # list of match dicts for preview
        self._propagation_snapshot = None  # (beads, color_idx) for undo

        # --- Viewers ---
        self.viewer = MolecularViewer(width=880, height=600)
        self.viewer.on_selection_change(self._on_selection_change)

        # --- CG Topology ---
        self.topology_manager = TopologyManager()
        self.cg_viewer = CGViewer(width=880, height=600)

        # --- Sidebar widgets ---
        self.file_input = pn.widgets.TextInput(
            name='Dump file path', placeholder='/path/to/file.dump',
            width=290,
        )
        self.load_btn = pn.widgets.Button(name='Load', button_type='primary', width=290)
        self.load_btn.on_click(self._on_load)

        self.hint_input = pn.widgets.TextInput(
            name='Atoms per molecule (optional)',
            placeholder='e.g. 12, 3',
            width=290,
        )

        self.status = pn.pane.Alert('Load a LAMMPS dump file to begin.', alert_type='info', width=290)

        # Format info display (populated after load)
        self.format_display = pn.pane.Markdown('', width=290)

        # Molecule type selector
        self.mol_selector = pn.widgets.Select(
            name='Molecule type', options=[], width=290, disabled=True,
        )
        self.mol_selector.param.watch(self._on_mol_select, 'value')

        # Mass assignment
        self.mass_header = pn.pane.Markdown('### Atom masses', width=290)
        self.mass_inputs = pn.Column(width=290)

        # Bead creation
        self.bead_name_input = pn.widgets.TextInput(
            name='Bead name', placeholder='e.g. BENZ', width=290,
        )
        self.create_bead_btn = pn.widgets.Button(
            name='Create Bead from Selection', button_type='success', width=290, disabled=True,
        )
        self.create_bead_btn.on_click(self._on_create_bead)

        self.clear_sel_btn = pn.widgets.Button(
            name='Clear Selection', button_type='warning', width=290,
        )
        self.clear_sel_btn.on_click(self._on_clear_selection)

        # --- Feature 1: Propagation preview panel ---
        self.propagate_preview = pn.pane.Markdown('', width=290)

        # --- Feature 2: Preview / Apply propagation ---
        self.preview_propagate_btn = pn.widgets.Button(
            name='Preview Propagation', button_type='default', width=290, disabled=True,
        )
        self.preview_propagate_btn.on_click(self._on_preview_propagate)

        self.apply_propagate_btn = pn.widgets.Button(
            name='Apply Propagation', button_type='success', width=290, disabled=True,
        )
        self.apply_propagate_btn.on_click(self._on_apply_propagate)

        # --- Feature 3: Highlight unassigned atoms ---
        self.show_unassigned_btn = pn.widgets.Toggle(
            name='Highlight Unassigned', button_type='warning', width=290,
        )
        self.show_unassigned_btn.param.watch(self._on_toggle_unassigned, 'value')

        # --- Feature 4: Undo propagation ---
        self.undo_propagate_btn = pn.widgets.Button(
            name='Undo Propagation', button_type='danger', width=290, disabled=True,
        )
        self.undo_propagate_btn.on_click(self._on_undo_propagate)

        # Selection display
        self.selection_display = pn.pane.Markdown('**Selected atoms:** none', width=290)

        # --- Feature 5: Improved beads display ---
        self.beads_display = pn.pane.Markdown('**Defined beads:** none', width=290)

        # Bead highlight selector
        self.bead_selector = pn.widgets.Select(
            name='Highlight bead', options={'(none)': ''}, width=290,
        )
        self.bead_selector.param.watch(self._on_bead_highlight, 'value')

        self.delete_bead_btn = pn.widgets.Button(
            name='Delete Selected Bead', button_type='danger', width=290,
        )
        self.delete_bead_btn.on_click(self._on_delete_bead)

        self.clear_template_btn = pn.widgets.Button(
            name='Clear All Template Beads', button_type='danger', width=290,
        )
        self.clear_template_btn.on_click(self._on_clear_template)

        # Export
        self.export_dir_input = pn.widgets.TextInput(
            name='Export directory', value='./cg_mapping_output', width=290,
        )
        self.export_btn = pn.widgets.Button(
            name='Export YAML', button_type='primary', width=290, disabled=True,
        )
        self.export_btn.on_click(self._on_export)
        self.export_status = pn.pane.Alert('', alert_type='info', visible=False, width=290)

        # --- Generate CG Trajectory widgets ---
        self.gen_format_selector = pn.widgets.Select(
            name='Output format', options=['xyz', 'npz', 'data'], value='npz', width=290,
        )
        self.gen_output_prefix = pn.widgets.TextInput(
            name='Output prefix', value='cg_output', width=290,
        )
        self.gen_wrap_checkbox = pn.widgets.Checkbox(
            name='Wrap coordinates', value=True, width=290,
        )
        self.gen_target_selector = pn.widgets.Select(
            name='Target', options=['all', 'coord'], value='all', width=290,
        )
        self.gen_frame_input = pn.widgets.IntInput(
            name='Frame index (data format only)', value=0, start=0, width=290,
        )
        self.gen_btn = pn.widgets.Button(
            name='Generate CG Trajectory', button_type='success', width=290, disabled=True,
        )
        self.gen_btn.on_click(self._on_generate)
        self.gen_status = pn.pane.Markdown('', width=290)

        # --- Bond visualization widgets ---
        self.show_bonds_checkbox = pn.widgets.Checkbox(name='Show bonds', value=True, width=290)
        self.show_bonds_checkbox.param.watch(self._on_toggle_bonds, 'value')

        self.data_file_input = pn.widgets.TextInput(
            name='LAMMPS data file (optional)', placeholder='/path/to/file.data', width=290)
        self.load_data_btn = pn.widgets.Button(name='Load Bonds', button_type='default', width=290)
        self.load_data_btn.on_click(self._on_load_data_file)

        self.bond_status = pn.pane.Markdown('*Bonds: auto-detect (distance)*', width=290)

        # --- CG Topology widgets ---
        self.infer_bonds_btn = pn.widgets.Button(
            name='Infer Bonds from Atomistic', button_type='primary', width=290, disabled=True,
        )
        self.infer_bonds_btn.on_click(self._on_infer_bonds)

        self.bond_editor_input = pn.widgets.TextInput(
            name='Bond input', placeholder='e.g. 0-1 or BENZ-WAT', width=290,
        )
        self.add_bond_btn = pn.widgets.Button(
            name='Add Bond', button_type='success', width=290,
        )
        self.add_bond_btn.on_click(self._on_add_bond)

        self.bond_list_select = pn.widgets.Select(
            name='Bond list', options={}, width=290,
        )
        self.bond_list_select.param.watch(self._on_bond_select, 'value')

        self.remove_bond_btn = pn.widgets.Button(
            name='Remove Selected Bond', button_type='danger', width=290,
        )
        self.remove_bond_btn.on_click(self._on_remove_bond)

        self.clear_all_bonds_btn = pn.widgets.Button(
            name='Clear All Bonds', button_type='danger', width=290,
        )
        self.clear_all_bonds_btn.on_click(self._on_clear_all_bonds)

        self.show_angles_toggle = pn.widgets.Toggle(
            name='Show Angles', value=False, width=290,
        )
        self.show_angles_toggle.param.watch(self._on_toggle_angles, 'value')

        self.show_dihedrals_toggle = pn.widgets.Toggle(
            name='Show Dihedrals', value=False, width=290,
        )
        self.show_dihedrals_toggle.param.watch(self._on_toggle_dihedrals, 'value')

        self.topology_summary = pn.pane.Markdown('', width=290)

    def _on_load(self, event):
        path = self.file_input.value.strip()
        if not path or not os.path.isfile(path):
            self.status.object = 'File not found. Please enter a valid path.'
            self.status.alert_type = 'danger'
            return

        self.status.object = 'Checking dump file format...'
        self.status.alert_type = 'info'
        self.format_display.object = ''

        fmt_info = check_dump_format(path)
        if not fmt_info['valid']:
            errors = '\n'.join(f'- {e}' for e in fmt_info['errors'])
            self.status.object = f"Invalid dump file format:\n{errors}"
            self.status.alert_type = 'danger'
            return

        # Show format info
        cols_str = ' '.join(fmt_info['columns'])
        lines = [f'**Trajectory format:**', f'Columns: `{cols_str}`']
        if fmt_info['has_coords']:
            lines.append('&#10003; Coordinates (x y z)')
        if fmt_info['has_forces']:
            lines.append('&#10003; Forces (fx fy fz)')
        if fmt_info['has_images']:
            lines.append('&#10003; Image flags (ix iy iz)')
        if fmt_info['has_mol']:
            lines.append('&#10003; Molecule IDs (mol)')
        if fmt_info['warnings']:
            for w in fmt_info['warnings']:
                lines.append(f'&#9888; {w}')
        self.format_display.object = '\n\n'.join(lines)

        self.status.object = 'Loading dump file...'
        self.status.alert_type = 'info'

        self.dump_data = read_dump_file(path)
        if not self.dump_data:
            self.status.object = 'Failed to read dump file.'
            self.status.alert_type = 'danger'
            return

        # Unwrap molecules across periodic boundaries
        topology_cache = None
        for time_step in self.dump_data:
            frame_data = self.dump_data[time_step]
            _, topology_cache = unwrap_frame(frame_data, topology_cache=topology_cache)

        first_ts = min(self.dump_data.keys())
        frame = self.dump_data[first_ts]

        detector = MoleculeDetector(frame)
        hint = self.hint_input.value.strip() or None
        self.mol_types = detector.detect(atoms_per_mol_hint=hint)

        if not self.mol_types:
            self.status.object = 'No molecule types detected.'
            self.status.alert_type = 'warning'
            return

        # Collect unique atom types across all molecule types
        all_types = set()
        for mt in self.mol_types:
            all_types.update(int(t) for t in mt.representative_types)

        # Build mass input widgets
        self.mass_inputs.clear()
        for t in sorted(all_types):
            w = pn.widgets.FloatInput(
                name=f'Type {t} mass', value=1.0, start=0.01, step=0.1, width=130,
            )
            self.mass_inputs.append(w)
            self.atom_masses[t] = w

        # Populate molecule selector
        options = {f"{mt.name} ({mt.atoms_per_mol} atoms x {mt.count})": mt.name
                   for mt in self.mol_types}
        self.mol_selector.options = options
        self.mol_selector.disabled = False
        self.mol_selector.value = list(options.values())[0]

        self.export_btn.disabled = False
        self.gen_btn.disabled = False
        self.create_bead_btn.disabled = False

        n_types = len(self.mol_types)
        self.status.object = f'Loaded {frame["num_atoms"]} atoms. Detected {n_types} molecule type(s).'
        self.status.alert_type = 'success'

        self._update_propagate_preview()

    def _on_mol_select(self, event):
        name = event.new
        if not name:
            return
        mt = self._get_mol_type(name)
        if mt is None:
            return
        self.current_mol_type = mt

        self._refresh_viewer()
        self._update_beads_display()
        self._update_selection_display()
        self._update_propagate_preview()
        self._refresh_cg_viewer()
        self._update_topology_display()

    def _on_selection_change(self, selected):
        self._update_selection_display()

    def _on_clear_selection(self, event):
        self.viewer.clear_selection()
        self._update_selection_display()

    def _on_create_bead(self, event):
        if self.current_mol_type is None:
            return
        name = self.bead_name_input.value.strip()
        if not name:
            self.status.object = 'Enter a bead name.'
            self.status.alert_type = 'warning'
            return

        selected = self.viewer.get_selected()
        if not selected:
            self.status.object = 'Select atoms first by clicking in the viewer.'
            self.status.alert_type = 'warning'
            return

        # Check for overlap with existing beads
        assigned = self.bead_manager.get_assigned_indices(self.current_mol_type.name)
        overlap = set(selected) & assigned
        if overlap:
            self.status.object = f'Atoms {sorted(overlap)} already assigned to another bead.'
            self.status.alert_type = 'danger'
            return

        # Check duplicate bead name
        existing_names = [b.name for b in self.bead_manager.get_beads(self.current_mol_type.name)]
        if name in existing_names:
            self.status.object = f'Bead "{name}" already exists for this molecule type.'
            self.status.alert_type = 'danger'
            return

        # Build x_weights from atom masses
        x_weights = []
        for idx in selected:
            atom_type = int(self.current_mol_type.representative_types[idx])
            mass_widget = self.atom_masses.get(atom_type)
            mass = mass_widget.value if mass_widget else 1.0
            x_weights.append(mass)

        f_weights = [1.0] * len(selected)

        bead = self.bead_manager.add_bead(
            self.current_mol_type.name, name, selected, x_weights, f_weights
        )

        self._refresh_viewer()

        self.bead_name_input.value = ''
        self._update_beads_display()
        self._update_selection_display()
        self._update_propagate_preview()
        self._refresh_cg_viewer()
        self._update_topology_display()

        # Validate bonds (remove any referencing invalid bead indices)
        self.topology_manager.validate_bonds(self.current_mol_type.name, len(self.bead_manager.get_beads(self.current_mol_type.name)))

        self.status.object = f'Created bead "{name}" with {len(selected)} atoms.'
        self.status.alert_type = 'success'

    def _on_load_data_file(self, event):
        path = self.data_file_input.value.strip()
        if not path or not os.path.isfile(path):
            self.bond_status.object = '*Error: file not found*'
            return
        try:
            bonds = parse_lammps_data_bonds(path)
            self.all_bonds_global = bonds
            self.bond_status.object = f'*Bonds: {len(bonds)} from data file*'
            self.preview_propagate_btn.disabled = False
            self.infer_bonds_btn.disabled = False
            self._refresh_viewer()
            self._update_propagate_preview()
        except Exception as e:
            self.bond_status.object = f'*Error reading bonds: {e}*'

    def _on_toggle_bonds(self, event):
        self.viewer.set_show_bonds(event.new)

    def _get_local_bonds(self):
        """Map global bonds to local indices of the current representative molecule."""
        if self.all_bonds_global is None or self.current_mol_type is None:
            return None
        mt = self.current_mol_type
        first_ts = min(self.dump_data.keys())
        frame = self.dump_data[first_ts]
        atom_ids = get_frame(frame, 'id').astype(int)
        # Atom IDs for this molecule's representative atoms
        start = mt.start_index
        mol_ids = atom_ids[start:start + mt.atoms_per_mol]
        id_set = set(int(aid) for aid in mol_ids)
        # Build mapping from global atom ID to local 0-indexed position
        id_to_local = {int(aid): idx for idx, aid in enumerate(mol_ids)}
        local_bonds = []
        for (ai, aj) in self.all_bonds_global:
            if ai in id_set and aj in id_set:
                local_bonds.append((id_to_local[ai], id_to_local[aj]))
        return local_bonds if local_bonds else None

    def _build_molecule_graph(self):
        """Build a NetworkX graph of the current molecule with fingerprinted nodes."""
        local_bonds = self._get_local_bonds()
        if local_bonds is None:
            return None
        mt = self.current_mol_type
        G = nx.Graph()
        for i in range(mt.atoms_per_mol):
            G.add_node(i, atom_type=int(mt.representative_types[i]))
        for ai, aj in local_bonds:
            G.add_edge(ai, aj)
        # Compute neighborhood fingerprints
        for node in G.nodes():
            own_type = G.nodes[node]['atom_type']
            neighbor_types = sorted(G.nodes[n]['atom_type'] for n in G.neighbors(node))
            second_neighbors = set()
            for n in G.neighbors(node):
                for nn in G.neighbors(n):
                    if nn != node:
                        second_neighbors.add(nn)
            second_neighbor_types = sorted(G.nodes[nn]['atom_type'] for nn in second_neighbors)
            G.nodes[node]['fingerprint'] = (own_type, tuple(neighbor_types), tuple(second_neighbor_types))
        return G

    def _find_propagation_matches(self):
        """Run the graph matcher and return list of match dicts without creating beads.

        Each match dict maps template_node -> full_graph_node (reverse iso).
        Returns (matches_list, template_beads) or ([], []) if not possible.
        """
        if self.all_bonds_global is None or self.current_mol_type is None:
            return [], []
        mt = self.current_mol_type
        beads = self.bead_manager.get_beads(mt.name)
        if not beads:
            return [], []

        G = self._build_molecule_graph()
        if G is None:
            return [], []

        # Use only template (non-propagated) bead indices to avoid exponential
        # graph matching when most atoms are already assigned.
        template_beads = [b for b in beads if not b.is_propagated]
        if not template_beads:
            return [], []
        template_indices = sorted(set(idx for b in template_beads for idx in b.atom_indices))
        if not template_indices:
            return [], []

        template_subgraph = G.subgraph(template_indices).copy()

        def fingerprint_match(n1_attrs, n2_attrs):
            return n1_attrs['atom_type'] == n2_attrs['atom_type']

        matcher = GraphMatcher(G, template_subgraph, node_match=fingerprint_match)

        already_assigned = set(self.bead_manager.get_assigned_indices(mt.name))
        matches = []

        for iso in matcher.subgraph_isomorphisms_iter():
            matched_full_nodes = set(iso.keys())
            if matched_full_nodes & already_assigned:
                continue
            rev_iso = {v: k for k, v in iso.items()}
            matches.append(rev_iso)
            already_assigned.update(matched_full_nodes)

        return matches, template_beads

    # --- Feature 1: Preview panel ---

    def _update_propagate_preview(self):
        """Update the propagation preview panel with template summary and match count."""
        if self.current_mol_type is None:
            self.propagate_preview.object = ''
            return

        mt = self.current_mol_type
        beads = self.bead_manager.get_beads(mt.name)
        template_beads = [b for b in beads if not b.is_propagated]
        if not template_beads:
            # All beads are propagated; show coverage and prompt for new template
            total_atoms = mt.atoms_per_mol
            assigned = self.bead_manager.get_assigned_indices(mt.name)
            pct = len(assigned) * 100 // total_atoms if total_atoms else 0
            self.propagate_preview.object = (
                f'**Propagation Preview**\n\n'
                f'Coverage: {len(assigned)} / {total_atoms} atoms ({pct}%)\n\n'
                f'*All current beads are propagated. Define new template beads to continue.*'
            )
            self.preview_propagate_btn.disabled = True
            return

        total_atoms = mt.atoms_per_mol
        assigned = self.bead_manager.get_assigned_indices(mt.name)
        template_atom_count = sum(len(b.atom_indices) for b in template_beads)

        lines = ['**Propagation Preview**']
        # Template composition
        bead_parts = [f'{b.name} ({len(b.atom_indices)})' for b in template_beads]
        lines.append(f'Template: {" + ".join(bead_parts)} = {template_atom_count} atoms')

        # Coverage
        pct = len(assigned) * 100 // total_atoms if total_atoms else 0
        lines.append(f'Coverage: {len(assigned)} / {total_atoms} atoms ({pct}%)')

        # Expected matches (count-only mode)
        if self.all_bonds_global is not None:
            matches, _ = self._find_propagation_matches()
            n_matches = len(matches)
            new_beads = n_matches * len(template_beads)
            lines.append(f'Found **{n_matches}** matching group(s) → will create **{new_beads}** new bead(s)')
        else:
            lines.append('*Load bonds to see match count*')

        self.propagate_preview.object = '\n\n'.join(lines)
        self.preview_propagate_btn.disabled = False

    # --- Feature 2: Preview / Apply propagation ---

    def _on_preview_propagate(self, event):
        """Run matching algorithm, highlight in viewer, but don't create beads."""
        if self.all_bonds_global is None:
            self.status.object = 'Load a LAMMPS data file first (bonds needed).'
            self.status.alert_type = 'warning'
            return
        if self.current_mol_type is None:
            self.status.object = 'Select a molecule type first.'
            self.status.alert_type = 'warning'
            return
        mt = self.current_mol_type
        beads = self.bead_manager.get_beads(mt.name)
        if not beads:
            self.status.object = 'Define at least one bead before previewing.'
            self.status.alert_type = 'warning'
            return

        matches, template_beads = self._find_propagation_matches()
        if not matches:
            self.status.object = 'No additional matching groups found.'
            self.status.alert_type = 'info'
            self._pending_propagation = []
            self.apply_propagate_btn.disabled = True
            return

        self._pending_propagation = matches

        # Build groups for viewer highlighting
        groups = []
        for rev_iso in matches:
            group_indices = list(rev_iso.values())
            groups.append(group_indices)

        self.viewer.highlight_groups(groups)
        self.apply_propagate_btn.disabled = False

        self.status.object = (
            f'{len(matches)} group(s) found. Click "Apply Propagation" to create beads.'
        )
        self.status.alert_type = 'info'

    def _on_apply_propagate(self, event):
        """Create beads from the previewed propagation matches."""
        if not self._pending_propagation or self.current_mol_type is None:
            return

        mt = self.current_mol_type
        beads = self.bead_manager.get_beads(mt.name)
        template_beads = [b for b in beads if not b.is_propagated]
        if not template_beads:
            template_beads = beads

        # Snapshot for undo
        self._propagation_snapshot = self.bead_manager.snapshot(mt.name)
        self.undo_propagate_btn.disabled = False

        name_counters = {}
        for b in template_beads:
            name_counters[b.name] = 1

        new_groups = 0
        for rev_iso in self._pending_propagation:
            for b in template_beads:
                if not all(idx in rev_iso for idx in b.atom_indices):
                    continue
                name_counters[b.name] += 1
                suffix = name_counters[b.name]
                new_name = f"{b.name}_{suffix}"
                new_indices = [rev_iso[idx] for idx in b.atom_indices]
                x_weights = []
                for idx in new_indices:
                    atom_type = int(mt.representative_types[idx])
                    mass_widget = self.atom_masses.get(atom_type)
                    mass = mass_widget.value if mass_widget else 1.0
                    x_weights.append(mass)
                f_weights = [1.0] * len(new_indices)
                self.bead_manager.add_bead(
                    mt.name, new_name, new_indices, x_weights, f_weights,
                    is_propagated=True, bead_type=b.bead_type,
                )
            new_groups += 1

        # Mark original template beads as propagated so new beads become the next template
        for b in template_beads:
            b.is_propagated = True

        self._pending_propagation = []
        self.apply_propagate_btn.disabled = True
        self.viewer.clear_group_highlights()
        self.viewer.clear_selection()

        self._refresh_viewer()
        self._update_beads_display()
        self._update_propagate_preview()

        unassigned = self.bead_manager.get_unassigned_indices(mt.name, mt.atoms_per_mol)
        self.status.object = f'Applied {new_groups} group(s). {len(unassigned)} atom(s) remain unassigned.'
        self.status.alert_type = 'success' if not unassigned else 'info'

    # --- Feature 3: Highlight unassigned atoms ---

    def _on_toggle_unassigned(self, event):
        """Toggle highlighting of unassigned atoms in the 3D viewer."""
        if event.new and self.current_mol_type is not None:
            mt = self.current_mol_type
            unassigned = self.bead_manager.get_unassigned_indices(mt.name, mt.atoms_per_mol)
            self.viewer.highlight_unassigned(unassigned)
        else:
            self.viewer.clear_unassigned_highlights()

    # --- Feature 4: Undo propagation ---

    def _on_undo_propagate(self, event):
        """Restore beads to the state before the last propagation."""
        if self._propagation_snapshot is None or self.current_mol_type is None:
            return

        mt = self.current_mol_type
        self.bead_manager.restore(mt.name, self._propagation_snapshot)
        self._propagation_snapshot = None
        self.undo_propagate_btn.disabled = True
        self._pending_propagation = []
        self.apply_propagate_btn.disabled = True

        self.viewer.clear_group_highlights()
        self._refresh_viewer()
        self._update_beads_display()
        self._update_propagate_preview()
        self._refresh_cg_viewer()
        self._update_topology_display()

        # Reset unassigned highlighting if active
        if self.show_unassigned_btn.value:
            unassigned = self.bead_manager.get_unassigned_indices(mt.name, mt.atoms_per_mol)
            self.viewer.highlight_unassigned(unassigned)

        self.status.object = 'Propagation undone.'
        self.status.alert_type = 'info'

    def _refresh_viewer(self):
        """Refresh the 3D viewer with current molecule and bond data."""
        if self.current_mol_type is None:
            return
        mt = self.current_mol_type
        highlights = {}
        for bead in self.bead_manager.get_beads(mt.name):
            for idx in bead.atom_indices:
                highlights[idx] = bead.color
        bonds = self._get_local_bonds()
        self.viewer.set_molecule(mt.representative_xyz, mt.representative_types, highlights, bonds=bonds)

        # Re-apply unassigned highlights if toggle is active
        if self.show_unassigned_btn.value:
            unassigned = self.bead_manager.get_unassigned_indices(mt.name, mt.atoms_per_mol)
            self.viewer.highlight_unassigned(unassigned)

    # --- CG Topology methods ---

    def _get_bead_name_by_index(self, idx):
        """Get bead name by index (position in bead list)."""
        if self.current_mol_type is None:
            return f'bead_{idx}'
        beads = self.bead_manager.get_beads(self.current_mol_type.name)
        if 0 <= idx < len(beads):
            return beads[idx].name
        return f'bead_{idx}'

    def _on_infer_bonds(self, event):
        """Infer CG bonds from atomistic connectivity."""
        if self.current_mol_type is None:
            self.status.object = 'Select a molecule type first.'
            self.status.alert_type = 'warning'
            return
        if self.all_bonds_global is None:
            self.status.object = 'Load a LAMMPS data file first (bonds needed).'
            self.status.alert_type = 'warning'
            return

        mt = self.current_mol_type
        beads = self.bead_manager.get_beads(mt.name)
        if not beads:
            self.status.object = 'Define at least one bead before inferring bonds.'
            self.status.alert_type = 'warning'
            return

        local_bonds = self._get_local_bonds()
        if not local_bonds:
            self.status.object = 'No local bonds found for this molecule type.'
            self.status.alert_type = 'warning'
            return

        count = self.topology_manager.infer_bonds_from_atomistic(mt.name, beads, local_bonds)
        self._refresh_cg_viewer()
        self._update_topology_display()
        self.status.object = f'Inferred {count} CG bond(s) from atomistic connectivity.'
        self.status.alert_type = 'success'

    def _on_add_bond(self, event):
        """Add a bond from the text input (index pair like '0-1' or name pair like 'BENZ-WAT')."""
        if self.current_mol_type is None:
            self.status.object = 'Select a molecule type first.'
            self.status.alert_type = 'warning'
            return

        text = self.bond_editor_input.value.strip()
        if not text:
            self.status.object = 'Enter a bond pair (e.g. "0-1" or "BENZ-WAT").'
            self.status.alert_type = 'warning'
            return

        parts = text.split('-', 1)
        if len(parts) != 2:
            self.status.object = 'Invalid format. Use "0-1" or "BENZ-WAT".'
            self.status.alert_type = 'danger'
            return

        mt = self.current_mol_type
        beads = self.bead_manager.get_beads(mt.name)
        a_str, b_str = parts[0].strip(), parts[1].strip()

        # Try parsing as indices first
        try:
            i, j = int(a_str), int(b_str)
        except ValueError:
            # Try as bead names
            name_to_idx = {b.name: idx for idx, b in enumerate(beads)}
            i = name_to_idx.get(a_str)
            j = name_to_idx.get(b_str)
            if i is None or j is None:
                self.status.object = f'Could not resolve bead names "{a_str}" or "{b_str}".'
                self.status.alert_type = 'danger'
                return

        if i < 0 or i >= len(beads) or j < 0 or j >= len(beads):
            self.status.object = f'Bead indices out of range (0-{len(beads)-1}).'
            self.status.alert_type = 'danger'
            return

        if i == j:
            self.status.object = 'Cannot create a bond between a bead and itself.'
            self.status.alert_type = 'danger'
            return

        result = self.topology_manager.add_bond(mt.name, i, j)
        if result is None:
            self.status.object = 'Bond already exists.'
            self.status.alert_type = 'warning'
            return

        self.bond_editor_input.value = ''
        self._refresh_cg_viewer()
        self._update_topology_display()
        name_i = self._get_bead_name_by_index(i)
        name_j = self._get_bead_name_by_index(j)
        self.status.object = f'Added bond: {name_i} — {name_j}'
        self.status.alert_type = 'success'

    def _on_bond_select(self, event):
        """Highlight the selected bond in the CG viewer."""
        val = event.new
        if not val:
            self.cg_viewer.clear_highlighted_bond()
            return
        # Value format: "i-j"
        try:
            parts = val.split('-')
            i, j = int(parts[0]), int(parts[1])
            self.cg_viewer.set_highlighted_bond(i, j)
        except (ValueError, IndexError):
            self.cg_viewer.clear_highlighted_bond()

    def _on_remove_bond(self, event):
        """Remove the bond currently selected in the bond list."""
        if self.current_mol_type is None:
            return
        val = self.bond_list_select.value
        if not val:
            self.status.object = 'Select a bond from the list first.'
            self.status.alert_type = 'warning'
            return
        try:
            parts = val.split('-')
            i, j = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            return

        if self.topology_manager.remove_bond(self.current_mol_type.name, i, j):
            self.cg_viewer.clear_highlighted_bond()
            self._refresh_cg_viewer()
            self._update_topology_display()
            name_i = self._get_bead_name_by_index(i)
            name_j = self._get_bead_name_by_index(j)
            self.status.object = f'Removed bond: {name_i} — {name_j}'
            self.status.alert_type = 'success'

    def _on_clear_all_bonds(self, event):
        """Clear all bonds for the current molecule type."""
        if self.current_mol_type is None:
            return
        self.topology_manager.clear_bonds(self.current_mol_type.name)
        self.cg_viewer.clear_highlighted_bond()
        self._refresh_cg_viewer()
        self._update_topology_display()
        self.status.object = 'Cleared all bonds.'
        self.status.alert_type = 'info'

    def _on_toggle_angles(self, event):
        """Toggle angle visualization."""
        self._refresh_cg_viewer()

    def _on_toggle_dihedrals(self, event):
        """Toggle dihedral visualization."""
        self._refresh_cg_viewer()

    def _refresh_cg_viewer(self):
        """Refresh the CG viewer with current beads and topology."""
        if self.current_mol_type is None:
            return
        mt = self.current_mol_type
        beads = self.bead_manager.get_beads(mt.name)
        if not beads:
            return

        # Compute bead positions
        bead_positions = compute_bead_positions(mt.representative_xyz, beads)
        bead_names = [b.name for b in beads]

        # Get bonds
        bonds = [(b.bead_i, b.bead_j) for b in self.topology_manager.get_bonds(mt.name)]

        # Get angles and dihedrals for visualization
        angles = self.topology_manager.infer_angles(mt.name) if self.show_angles_toggle.value else []
        dihedrals = self.topology_manager.infer_dihedrals(mt.name) if self.show_dihedrals_toggle.value else []

        self.cg_viewer.set_beads(
            bead_positions, bead_names,
            bonds=bonds,
            angles=angles,
            dihedrals=dihedrals
        )

    def _update_topology_display(self):
        """Update the topology display with current bonds and summary."""
        if self.current_mol_type is None:
            self.bond_list_select.options = {}
            self.bond_list_select.value = None
            self.topology_summary.object = ''
            return

        mt = self.current_mol_type
        bonds = self.topology_manager.get_bonds(mt.name)

        if not bonds:
            self.bond_list_select.options = {}
            self.bond_list_select.value = None
        else:
            options = {}
            for bond in bonds:
                name_i = self._get_bead_name_by_index(bond.bead_i)
                name_j = self._get_bead_name_by_index(bond.bead_j)
                label = f'{bond.bead_i}: {name_i} — {name_j}'
                value = f'{bond.bead_i}-{bond.bead_j}'
                options[label] = value
            self.bond_list_select.options = options
            self.bond_list_select.value = None

        # Update summary
        summary = self.topology_manager.get_summary(mt.name)
        self.topology_summary.object = (
            f'**Topology summary:** {summary["bonds"]} bonds, '
            f'{summary["angles"]} angles, {summary["dihedrals"]} dihedrals'
        )

    def _on_export(self, event):
        try:
            out_dir = self.export_dir_input.value.strip()
            if not out_dir:
                self.export_status.object = 'Enter an export directory.'
                self.export_status.alert_type = 'warning'
                self.export_status.visible = True
                return

            beads_by_mol = {
                mt.name: self.bead_manager.get_beads(mt.name)
                for mt in self.mol_types
            }

            if not any(beads_by_mol.values()):
                self.export_status.object = 'Define at least one bead before exporting.'
                self.export_status.alert_type = 'warning'
                self.export_status.visible = True
                return

            path = export_system(self.mol_types, beads_by_mol, out_dir,
                               topology_manager=self.topology_manager)
            self.export_status.object = f'Exported to {os.path.abspath(out_dir)}'
            self.export_status.alert_type = 'success'
            self.export_status.visible = True
        except Exception as e:
            self.export_status.object = f'Export failed: {e}'
            self.export_status.alert_type = 'danger'
            self.export_status.visible = True

    def _on_generate(self, event):
        out_dir = self.export_dir_input.value.strip()
        if not out_dir:
            self.status.object = 'Enter an export directory.'
            self.status.alert_type = 'warning'
            return

        beads_by_mol = {
            mt.name: self.bead_manager.get_beads(mt.name)
            for mt in self.mol_types
        }

        if not any(beads_by_mol.values()):
            self.status.object = 'Define at least one bead before generating.'
            self.status.alert_type = 'warning'
            return

        fmt = self.gen_format_selector.value
        prefix = self.gen_output_prefix.value.strip() or 'cg_output'
        do_wrap = self.gen_wrap_checkbox.value
        target = self.gen_target_selector.value
        frame_idx = self.gen_frame_input.value

        try:
            self.gen_status.object = 'Exporting YAML files...'
            self.status.object = 'Generating CG trajectory...'
            self.status.alert_type = 'info'

            # 1. Export YAML to output dir (reuse existing logic)
            system_yaml_path = export_system(self.mol_types, beads_by_mol, out_dir,
                                           topology_manager=self.topology_manager)

            # 2. Read back system_data, fixing paths to be absolute
            system_data = read_mapping_file(system_yaml_path)
            abs_names = []
            for name in system_data['system']['names']:
                abs_names.append(os.path.join(os.path.abspath(out_dir), name))
            system_data['system']['names'] = abs_names

            # 3. Loop frames and run cg_map
            cg_data = {'R': {}, 'F': {}, 'z': {}, 'cell': {}}
            n_frames = len(self.dump_data)
            for index, time_step in enumerate(self.dump_data.keys()):
                self.gen_status.object = f'Processing frame {index + 1}/{n_frames}...'
                frame_data = self.dump_data[time_step]
                cg_site, cg_coord, cg_force, cg_box = cg_map(
                    frame_data, system_data, target=target,
                )
                if do_wrap:
                    cg_coord = wrap_coordinates(cg_coord, cg_box)
                cg_data['R'][index] = cg_coord
                cg_data['z'][index] = cg_site
                cg_data['F'][index] = cg_force
                cg_data['cell'][index] = cg_box

            # 4. Prepare topology for LAMMPS data export
            topology_for_export = []
            if fmt == 'data':
                for mt in self.mol_types:
                    beads = beads_by_mol.get(mt.name, [])
                    if not beads:
                        continue
                    bonds_list = [(b.bead_i, b.bead_j, b.bond_type)
                                  for b in self.topology_manager.get_bonds(mt.name)]
                    raw_angles = self.topology_manager.infer_angles(mt.name)
                    raw_dihedrals = self.topology_manager.infer_dihedrals(mt.name)

                    # Assign angle types based on bead-type combinations
                    bead_types = [b.bead_type for b in beads]
                    angle_type_map = {}
                    angles_list = []
                    for (i, j, k) in raw_angles:
                        key = (bead_types[i], bead_types[j], bead_types[k])
                        # Use canonical (non-directional) key
                        canon_key = min(key, key[::-1])
                        if canon_key not in angle_type_map:
                            angle_type_map[canon_key] = len(angle_type_map) + 1
                        angles_list.append((i, j, k, angle_type_map[canon_key]))

                    # Assign dihedral types based on bead-type combinations
                    dihedral_type_map = {}
                    dihedrals_list = []
                    for (i, j, k, l) in raw_dihedrals:
                        key = (bead_types[i], bead_types[j], bead_types[k], bead_types[l])
                        canon_key = min(key, key[::-1])
                        if canon_key not in dihedral_type_map:
                            dihedral_type_map[canon_key] = len(dihedral_type_map) + 1
                        dihedrals_list.append((i, j, k, l, dihedral_type_map[canon_key]))

                    if bonds_list or angles_list or dihedrals_list:
                        topology_for_export.append({
                            'bonds': bonds_list,
                            'angles': angles_list,
                            'dihedrals': dihedrals_list,
                            'beads_per_mol': len(beads),
                            'n_molecules': mt.count,
                        })

            # 5. Write output
            out_prefix = os.path.join(os.path.abspath(out_dir), prefix)
            if fmt == 'xyz':
                write_xyz_trajectory(f'{out_prefix}.xyz', cg_data)
                out_file = f'{out_prefix}.xyz'
            elif fmt == 'npz':
                save_cg_data_npz(f'{out_prefix}.npz', cg_data)
                out_file = f'{out_prefix}.npz'
            elif fmt == 'data':
                write_lammps_data(f'{out_prefix}.data', cg_data, frame_idx=frame_idx,
                                topology=topology_for_export if topology_for_export else None)
                out_file = f'{out_prefix}.data'

            self.gen_status.object = f'**Done!** {n_frames} frame(s) written to `{out_file}`'
            self.status.object = f'CG trajectory generated: {out_file}'
            self.status.alert_type = 'success'

        except Exception as e:
            self.gen_status.object = f'**Error:** {e}'
            self.status.object = f'Generation failed: {e}'
            self.status.alert_type = 'danger'

    def _update_selection_display(self):
        selected = self.viewer.get_selected()
        if selected:
            self.selection_display.object = f'**Selected atoms:** {selected}'
        else:
            self.selection_display.object = '**Selected atoms:** none'

    # --- Feature 5: Improved beads display ---

    def _update_beads_display(self):
        if self.current_mol_type is None:
            self.beads_display.object = '**Defined beads:** none'
            self.bead_selector.options = {'(none)': ''}
            self.bead_selector.value = ''
            return

        beads = self.bead_manager.get_beads(self.current_mol_type.name)
        if not beads:
            self.beads_display.object = '**Defined beads:** none'
            self.bead_selector.options = {'(none)': ''}
            self.bead_selector.value = ''
            return

        template_beads = [b for b in beads if not b.is_propagated]
        propagated_beads = [b for b in beads if b.is_propagated]

        lines = []

        # Template beads section
        if template_beads:
            lines.append('**Template beads:**')
            for b in template_beads:
                idx_str = self._format_indices(b.atom_indices)
                lines.append(
                    f'- <span style="color:{b.color}">&#9679;</span> **{b.name}**: '
                    f'atoms [{idx_str}] ({len(b.atom_indices)} atoms)'
                )

        # Propagated beads section
        if propagated_beads:
            # Group propagated beads by match group
            # Beads from same propagation group share contiguous indices
            groups = self._group_propagated_beads(propagated_beads)
            lines.append(f'**Propagated ({len(groups)} group(s)):**')
            for group in groups:
                names = ', '.join(b.name for b in group)
                all_indices = []
                for b in group:
                    all_indices.extend(b.atom_indices)
                idx_str = self._format_indices(sorted(all_indices))
                color = group[0].color
                lines.append(
                    f'- <span style="color:{color}">&#9679;</span> {names}: atoms [{idx_str}]'
                )

        # Coverage bar
        total = self.current_mol_type.atoms_per_mol
        assigned = self.bead_manager.get_assigned_indices(self.current_mol_type.name)
        n_assigned = len(assigned)
        pct = n_assigned * 100 // total if total else 0
        filled = pct // 5  # 20 chars total
        bar = '\u2588' * filled + '\u2591' * (20 - filled)
        lines.append(f'\nCoverage: {bar} {pct}% ({n_assigned}/{total} atoms)')

        unassigned = self.bead_manager.get_unassigned_indices(
            self.current_mol_type.name, total
        )
        if unassigned:
            lines.append(f'*Unassigned: {self._format_indices(unassigned)}*')

        self.beads_display.object = '\n\n'.join(lines)

        # Update bead selector options
        options = {'(none)': ''}
        for b in beads:
            options[b.name] = b.name
        self.bead_selector.options = options
        self.bead_selector.value = ''

    @staticmethod
    def _format_indices(indices):
        """Format a sorted list of indices as compact ranges (e.g. '0-5, 8, 10-12')."""
        if not indices:
            return ''
        indices = sorted(indices)
        ranges = []
        start = indices[0]
        end = indices[0]
        for i in indices[1:]:
            if i == end + 1:
                end = i
            else:
                ranges.append(f'{start}-{end}' if end > start else str(start))
                start = end = i
        ranges.append(f'{start}-{end}' if end > start else str(start))
        return ', '.join(ranges)

    @staticmethod
    def _group_propagated_beads(propagated_beads):
        """Group propagated beads by their suffix number (e.g. BENZ_2 and LINK_2 are one group)."""
        groups_dict = {}
        for b in propagated_beads:
            # Extract suffix: "NAME_N" -> N
            parts = b.name.rsplit('_', 1)
            suffix = parts[1] if len(parts) == 2 and parts[1].isdigit() else b.name
            if suffix not in groups_dict:
                groups_dict[suffix] = []
            groups_dict[suffix].append(b)
        return list(groups_dict.values())

    def _on_delete_bead(self, event):
        bead_name = self.bead_selector.value
        if not bead_name:
            self.status.object = 'Select a bead to delete from the dropdown.'
            self.status.alert_type = 'warning'
            return
        if self.current_mol_type is None:
            return
        mt = self.current_mol_type
        self.bead_manager.remove_bead(mt.name, bead_name)
        self._pending_propagation = []
        self._propagation_snapshot = None
        self.undo_propagate_btn.disabled = True
        self.apply_propagate_btn.disabled = True
        self._refresh_viewer()
        self._update_beads_display()
        self._update_propagate_preview()
        self._refresh_cg_viewer()
        self._update_topology_display()

        # Validate bonds
        self.topology_manager.validate_bonds(mt.name, len(self.bead_manager.get_beads(mt.name)))

        self.status.object = f'Deleted bead "{bead_name}".'
        self.status.alert_type = 'success'

    def _on_clear_template(self, event):
        if self.current_mol_type is None:
            self.status.object = 'Select a molecule type first.'
            self.status.alert_type = 'warning'
            return
        mt = self.current_mol_type
        beads = self.bead_manager.get_beads(mt.name)
        if not beads:
            self.status.object = 'No beads to clear.'
            self.status.alert_type = 'warning'
            return
        self.bead_manager.clear_beads(mt.name)
        self._pending_propagation = []
        self._propagation_snapshot = None
        self.undo_propagate_btn.disabled = True
        self.apply_propagate_btn.disabled = True
        self._refresh_viewer()
        self._update_beads_display()
        self._update_propagate_preview()
        self._refresh_cg_viewer()
        self._update_topology_display()

        # Clear bonds too
        self.topology_manager.clear_bonds(mt.name)

        self.status.object = 'Cleared all beads.'
        self.status.alert_type = 'success'

    def _on_bead_highlight(self, event):
        """Highlight atoms of the selected bead in the 3D viewer."""
        bead_name = event.new
        if not bead_name or self.current_mol_type is None:
            self.viewer.clear_highlight()
            return
        for b in self.bead_manager.get_beads(self.current_mol_type.name):
            if b.name == bead_name:
                self.viewer.highlight_atoms(b.atom_indices)
                return
        self.viewer.clear_highlight()

    def _get_mol_type(self, name):
        for mt in self.mol_types:
            if mt.name == name:
                return mt
        return None

    def layout(self):
        # ========== TOP HEADER - Centered with handwritten style ==========
        header = pn.pane.HTML(
            '''<div style="
                width: 100%;
                text-align: center;
                padding: 20px 0 16px 0;
                background: linear-gradient(180deg, #FFFFFF 0%, #FAFBFC 100%);
                border-bottom: 1px solid #E2E8F0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.02);
            ">
                <div style="
                    font-family: 'Caveat', cursive, -apple-system, sans-serif;
                    font-size: 2.5rem;
                    font-weight: 700;
                    color: #1A1A2E;
                    letter-spacing: 0.02em;
                    line-height: 1;
                    margin-bottom: 4px;
                ">CG-Map</div>
                <div style="
                    font-size: 0.8125rem;
                    color: #64748B;
                    font-weight: 500;
                    letter-spacing: 0.03em;
                ">Coarse-Grained Molecular Mapping</div>
            </div>''',
            sizing_mode='stretch_width',
            height=90,
        )
        
        # ========== LEFT SIDEBAR: Core Controls ==========
        left_sidebar = pn.Column(
            pn.pane.Markdown('## Load'),
            self.file_input,
            self.hint_input,
            self.load_btn,
            self.format_display,
            pn.layout.Divider(),
            self.mass_header,
            self.mass_inputs,
            pn.layout.Divider(),
            pn.pane.Markdown('## Molecule'),
            self.mol_selector,
            pn.layout.Divider(),
            pn.pane.Markdown('## Bonds'),
            self.show_bonds_checkbox,
            self.data_file_input,
            self.load_data_btn,
            self.bond_status,
            pn.layout.Divider(),
            pn.pane.Markdown('## Create Bead'),
            self.selection_display,
            self.bead_name_input,
            self.create_bead_btn,
            self.clear_sel_btn,
            pn.layout.Divider(),
            self.status,
            width=380,
            sizing_mode='fixed',
            css_classes=['left-sidebar'],
            styles={'background': '#FFFFFF', 'border-radius': '14px', 'padding': '20px'},
            margin=(16, 0, 16, 24),
        )

        # ========== RIGHT SIDEBAR: Advanced Controls ==========
        right_sidebar = pn.Column(
            pn.pane.Markdown('## Propagation'),
            self.propagate_preview,
            self.preview_propagate_btn,
            self.apply_propagate_btn,
            self.undo_propagate_btn,
            self.show_unassigned_btn,
            pn.layout.Divider(),
            pn.pane.Markdown('## Beads'),
            self.beads_display,
            self.bead_selector,
            self.delete_bead_btn,
            self.clear_template_btn,
            pn.layout.Divider(),
            pn.pane.Markdown('## Topology'),
            self.infer_bonds_btn,
            self.bond_editor_input,
            self.add_bond_btn,
            self.bond_list_select,
            self.remove_bond_btn,
            self.clear_all_bonds_btn,
            self.show_angles_toggle,
            self.show_dihedrals_toggle,
            self.topology_summary,
            width=380,
            sizing_mode='fixed',
            css_classes=['right-sidebar'],
            styles={'background': '#FFFFFF', 'border-radius': '14px', 'padding': '20px'},
            margin=(16, 24, 16, 0),
        )

        # ========== CENTER: Visualization + Export/Generate ==========
        atomistic_tab = pn.Column(
            self.viewer.get_panel(),
            sizing_mode='fixed',
            margin=0,
        )

        cg_tab = pn.Column(
            self.cg_viewer.get_panel(),
            sizing_mode='fixed',
            margin=0,
        )

        viewer_tabs = pn.Tabs(
            ('Atomistic', atomistic_tab),
            ('CG', cg_tab),
            sizing_mode='fixed',
            width=920,
            styles={'background': '#FFFFFF', 'border-radius': '14px', 'box-shadow': '0 4px 6px -1px rgba(0,0,0,0.05)'},
        )

        export_section = pn.Column(
            pn.pane.Markdown('## Export'),
            pn.Row(
                pn.Column(self.export_dir_input, width=520),
                pn.Column(self.export_btn, self.export_status, width=200),
                sizing_mode='fixed',
            ),
            width=920,
            sizing_mode='fixed',
            styles={'background': '#FFFFFF', 'border-radius': '14px', 'padding': '16px 20px', 'margin-top': '12px'},
        )

        generate_section = pn.Column(
            pn.pane.Markdown('## Generate CG Trajectory'),
            pn.Row(
                pn.Column(
                    pn.Row(self.gen_format_selector, self.gen_output_prefix, sizing_mode='fixed'),
                    sizing_mode='fixed',
                ),
                pn.Column(
                    pn.Row(self.gen_wrap_checkbox, self.gen_target_selector, sizing_mode='fixed'),
                    sizing_mode='fixed',
                ),
                pn.Column(
                    pn.Row(self.gen_frame_input, self.gen_btn, sizing_mode='fixed'),
                    sizing_mode='fixed',
                ),
                sizing_mode='fixed',
            ),
            self.gen_status,
            width=920,
            sizing_mode='fixed',
            styles={'background': '#FFFFFF', 'border-radius': '14px', 'padding': '16px 20px', 'margin-top': '12px'},
        )

        center_column = pn.Column(
            viewer_tabs,
            export_section,
            generate_section,
            sizing_mode='fixed',
            width=920,
            margin=(16, 16, 16, 16),
        )

        # ========== MAIN CONTENT AREA ==========
        main_content = pn.Row(
            left_sidebar,
            pn.Spacer(width=16),
            center_column,
            pn.Spacer(width=16),
            right_sidebar,
            sizing_mode='stretch_width',
            styles={'background': '#FAFBFC', 'justify-content': 'center'},
        )
        
        return pn.Column(header, main_content, sizing_mode='stretch_both', styles={'background': '#FAFBFC'})


def create_app():
    """Create and return the Panel app."""
    app = CGMapApp()
    return app.layout()


def launch_app(port=5006, show=True):
    """Launch the GUI server."""
    pn.serve(create_app, port=port, show=show, title='CG Mapping GUI')


def _cli_main():
    """CLI entry point for cgmap-gui command."""
    import argparse
    parser = argparse.ArgumentParser(description='CG Mapping GUI')
    parser.add_argument('--port', type=int, default=5006, help='Port (default: 5006)')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser')
    args = parser.parse_args()
    launch_app(port=args.port, show=not args.no_browser)
