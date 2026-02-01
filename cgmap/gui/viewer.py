import html
import numpy as np
import param
import panel as pn


class ClickBridge(pn.reactive.ReactiveHTML):
    """Hidden bridge: listens for postMessage from 3Dmol iframe, syncs to Python."""
    click_value = param.String(default='')
    camera_value = param.String(default='')

    _template = '<div id="bridge" style="display:none">${click_value}</div>'

    _scripts = {
        'render': """
            if (!window._cgmapClickListener) {
                window._cgmapClickListener = true;
                window.addEventListener('message', function(evt) {
                    var d = evt.data;
                    if (d && d.type === 'cgmap-click') {
                        data.click_value = d.value;
                    } else if (d && d.type === 'cgmap-camera') {
                        data.camera_value = d.value;
                    }
                });
            }
        """
    }


# Element symbols by LAMMPS type (common defaults)
DEFAULT_ELEMENTS = {1: 'C', 2: 'H', 3: 'O', 4: 'N', 5: 'S', 6: 'F'}

# CPK colors by element
ELEMENT_COLORS = {
    'C': '#909090', 'H': '#FFFFFF', 'O': '#FF0D0D', 'N': '#3050F8',
    'S': '#FFFF30', 'F': '#90E050', 'P': '#FF8000', 'Cl': '#1FF01F',
}

# Colors by LAMMPS atom type index
TYPE_COLORS = [
    '#64748B', '#3B82F6', '#10B981', '#8B5CF6', '#F59E0B',
    '#EC4899', '#06B6D4', '#F97316', '#6366F1', '#84CC16',
]


class MolecularViewer:
    """3Dmol.js viewer wrapped in a Panel HTML pane with click-to-select."""

    # Alternating colors for group highlighting
    GROUP_COLORS = [
        '#00BFFF', '#FF6347', '#32CD32', '#FFD700', '#FF69B4',
        '#00CED1', '#FF8C00', '#9370DB', '#20B2AA', '#DC143C',
    ]
    UNASSIGNED_COLOR = '#FF0000'

    def __init__(self, width=600, height=400):
        self.width = width
        self.height = height
        self._xyz = None
        self._types = None
        self._selected = set()
        self._highlight_indices = set()
        self._bead_highlights = {}  # index -> color
        self._group_highlights = {}  # index -> color (for propagation preview)
        self._unassigned_highlights = set()  # indices to show as unassigned
        self._bonds = None       # list of (i,j) 0-indexed tuples, or None for auto-detect
        self._show_bonds = True  # toggle
        self._needs_zoom_reset = True  # zoomTo() on new molecule, preserve view on selection changes
        self._saved_camera = None  # last known camera state JSON string

        # ReactiveHTML bridge for JS -> Python communication
        self._click_bridge = ClickBridge()
        self._click_bridge.param.watch(self._on_click, 'click_value')
        self._click_bridge.param.watch(self._on_camera, 'camera_value')

        self._html_pane = pn.pane.HTML(
            self._empty_html(),
            width=self.width,
            height=self.height + 20,
            sizing_mode='fixed',
        )

        self._selection_callbacks = []

    def on_selection_change(self, callback):
        """Register callback(selected_set) for selection changes."""
        self._selection_callbacks.append(callback)

    def get_panel(self):
        """Return the Panel component (HTML pane + hidden input)."""
        return pn.Column(self._html_pane, self._click_bridge, sizing_mode='fixed',
                         height=self.height + 30)

    def set_molecule(self, xyz, types, bead_highlights=None, bonds=None):
        """Render a molecule.

        Args:
            xyz: (N, 3) array of coordinates.
            types: (N,) array of atom type integers.
            bead_highlights: Optional dict {atom_index: color_hex}.
            bonds: Optional list of (i,j) 0-indexed tuples, or None for auto-detect.
        """
        self._xyz = np.array(xyz)
        self._types = np.array(types, dtype=int)
        self._selected = set()
        self._highlight_indices = set()
        self._bead_highlights = bead_highlights or {}
        self._group_highlights = {}
        self._unassigned_highlights = set()
        self._bonds = bonds
        self._needs_zoom_reset = True
        self._update_html()

    def set_bonds(self, bonds):
        """Set explicit bonds (list of (i,j) 0-indexed), or None for auto-detect."""
        self._bonds = bonds
        self._needs_zoom_reset = False
        self._update_html()

    def set_show_bonds(self, show):
        self._show_bonds = show
        self._needs_zoom_reset = False
        self._update_html()

    def set_selection(self, indices):
        """Set selected atom indices programmatically."""
        self._selected = set(indices)
        self._needs_zoom_reset = False
        self._update_html()

    def clear_selection(self):
        self._selected = set()
        self._needs_zoom_reset = False
        self._update_html()

    def highlight_atoms(self, indices):
        """Highlight atoms (for bead dropdown) without affecting click selection."""
        self._highlight_indices = set(indices)
        self._needs_zoom_reset = False
        self._update_html()

    def clear_highlight(self):
        """Clear bead highlight without affecting click selection."""
        self._highlight_indices = set()
        self._needs_zoom_reset = False
        self._update_html()

    def highlight_groups(self, groups):
        """Highlight multiple atom groups with alternating colors.

        Args:
            groups: list of lists of atom indices, one per matched group.
        """
        self._group_highlights = {}
        for gi, group_indices in enumerate(groups):
            color = self.GROUP_COLORS[gi % len(self.GROUP_COLORS)]
            for idx in group_indices:
                self._group_highlights[idx] = color
        self._needs_zoom_reset = False
        self._update_html()

    def clear_group_highlights(self):
        self._group_highlights = {}
        self._needs_zoom_reset = False
        self._update_html()

    def highlight_unassigned(self, indices):
        """Highlight unassigned atoms in red with larger radius."""
        self._unassigned_highlights = set(indices)
        self._needs_zoom_reset = False
        self._update_html()

    def clear_unassigned_highlights(self):
        self._unassigned_highlights = set()
        self._needs_zoom_reset = False
        self._update_html()

    def get_selected(self):
        return sorted(self._selected)

    def _on_camera(self, event):
        val = event.new.strip()
        if val:
            self._saved_camera = val

    def _on_click(self, event):
        val = event.new.strip()
        if not val:
            return
        try:
            # Value format: "idx_timestamp" to ensure change detection
            idx = int(val.split('_')[0])
        except (ValueError, IndexError):
            return

        if idx in self._selected:
            self._selected.discard(idx)
        else:
            self._selected.add(idx)

        self._needs_zoom_reset = False
        self._update_html()

        for cb in self._selection_callbacks:
            cb(set(self._selected))

    def _empty_html(self):
        doc = f"""<!DOCTYPE html>
<html><head><style>body{{margin:0;overflow:hidden}}</style></head>
<body><div style="width:{self.width}px;height:{self.height}px;background:linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);display:flex;align-items:center;justify-content:center;color:#64748B;font-family:sans-serif;">Load a dump file to view molecules</div></body></html>"""
        return f'<iframe srcdoc="{html.escape(doc, quote=True)}" style="width:{self.width}px;height:{self.height}px;border:none;"></iframe>'

    def _update_html(self):
        if self._xyz is None:
            self._html_pane.object = self._empty_html()
            return
        self._html_pane.object = self._build_html()

    def _atom_color(self, idx):
        if idx in self._selected or idx in self._highlight_indices:
            return '#FFD700'  # gold for selection or highlight
        if idx in self._unassigned_highlights:
            return self.UNASSIGNED_COLOR
        if idx in self._group_highlights:
            return self._group_highlights[idx]
        if idx in self._bead_highlights:
            return self._bead_highlights[idx]
        type_val = int(self._types[idx])
        return TYPE_COLORS[type_val % len(TYPE_COLORS)]

    def _atom_radius(self, idx):
        if idx in self._selected or idx in self._highlight_indices:
            return 0.5
        if idx in self._unassigned_highlights:
            return 0.55
        if idx in self._group_highlights:
            return 0.45
        return 0.35

    def _build_html(self):
        # Build XYZ string for 3Dmol
        n = len(self._xyz)
        lines = [f"{n}", "molecule"]
        for i in range(n):
            t = int(self._types[i])
            elem = DEFAULT_ELEMENTS.get(t, 'X')
            x, y, z = self._xyz[i]
            lines.append(f"{elem} {x:.6f} {y:.6f} {z:.6f}")
        xyz_str = "\\n".join(lines)

        # Determine bond rendering mode
        auto_detect = self._show_bonds and self._bonds is None
        explicit_bonds = self._show_bonds and self._bonds is not None

        # addModel options
        add_model_opts = ', {assignBonds: true}' if auto_detect else ''

        # Build per-atom style specs
        style_specs = []
        for i in range(n):
            color = self._atom_color(i)
            radius = self._atom_radius(i)
            if auto_detect:
                style_specs.append(
                    f'viewer.setStyle({{index:{i}}}, {{sphere:{{radius:{radius},color:"{color}"}}, stick:{{radius:0.08,color:"{color}"}}}});'
                )
            else:
                style_specs.append(
                    f'viewer.setStyle({{index:{i}}}, {{sphere:{{radius:{radius},color:"{color}"}}}});'
                )
        styles_js = "\n".join(style_specs)

        # Build explicit bond cylinders
        cylinder_specs = []
        if explicit_bonds:
            for (i, j) in self._bonds:
                if i >= n or j >= n:
                    continue
                xi, yi, zi = self._xyz[i]
                xj, yj, zj = self._xyz[j]
                ci = self._atom_color(i)
                cj = self._atom_color(j)
                cylinder_specs.append(
                    f'viewer.addCylinder({{start:{{x:{xi:.4f},y:{yi:.4f},z:{zi:.4f}}},end:{{x:{xj:.4f},y:{yj:.4f},z:{zj:.4f}}},radius:0.08,fromCap:1,toCap:1,color:"{ci}",color2:"{cj}"}});'
                )
        cylinders_js = "\n".join(cylinder_specs)

        # Build label specs for selected and highlighted atoms
        label_specs = []
        label_indices = sorted(self._selected | self._highlight_indices)
        for i in label_indices:
            t = int(self._types[i])
            label_specs.append(
                f'viewer.addLabel("{i}(T{t})", {{position:{{x:{self._xyz[i][0]:.4f},y:{self._xyz[i][1]:.4f},z:{self._xyz[i][2]:.4f}}},fontSize:10,backgroundColor:"rgba(0,0,0,0.6)",fontColor:"#FFD700",showBackground:true}});'
            )
        labels_js = "\n".join(label_specs)

        viewer_id = "viewer_container"

        # Build a self-contained HTML document for the iframe
        inner_doc = f"""<!DOCTYPE html>
<html>
<head>
<script src="https://3dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
<style>body{{margin:0;overflow:hidden}}</style>
</head>
<body>
<div id="{viewer_id}" style="width:{self.width}px;height:{self.height}px;position:relative;" oncontextmenu="return false;"></div>
<script>
(function() {{
    var element = document.getElementById("{viewer_id}");
    var viewer = $3Dmol.createViewer(element, {{backgroundColor: "0xF8FAFC"}});
    var xyz = "{xyz_str}";
    viewer.addModel(xyz, "xyz"{add_model_opts});
    {styles_js}
    {cylinders_js}
    {labels_js}
    viewer.setClickable({{}}, true, function(atom) {{
        var idx = atom.index;
        var val = '' + idx + '_' + Date.now();
        console.log('cgmap: atom clicked, index=' + idx);
        window.parent.postMessage({{type: 'cgmap-click', value: val}}, '*');
        console.log('cgmap: posted message, val=' + val);
    }});

    var needsZoom = {'true' if self._needs_zoom_reset else 'false'};
    var savedCamera = {self._saved_camera if self._saved_camera else 'null'};
    if (needsZoom) {{
        viewer.zoomTo();
    }} else if (savedCamera) {{
        viewer.setView(savedCamera);
    }} else {{
        viewer.zoomTo();
    }}
    viewer.render();

    setInterval(function() {{
        try {{
            var pos = viewer.getView();
            window.parent.postMessage({{type:'cgmap-camera', value: JSON.stringify(pos)}}, '*');
        }} catch(e) {{}}
    }}, 200);
}})();
</script>
</body>
</html>"""

        iframe = f'<iframe srcdoc="{html.escape(inner_doc, quote=True)}" style="width:{self.width}px;height:{self.height}px;border:none;"></iframe>'
        return iframe
