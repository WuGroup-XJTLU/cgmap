"""3Dmol.js viewer for coarse-grained beads."""

import html
import numpy as np
import panel as pn


def compute_bead_positions(representative_xyz: np.ndarray, beads: list) -> np.ndarray:
    """Compute bead center-of-mass positions from atomistic coordinates.

    Args:
        representative_xyz: (N, 3) array of atom coordinates.
        beads: List of BeadDefinition objects.

    Returns:
        (M, 3) array of bead positions where M = len(beads).
    """
    positions = []
    for bead in beads:
        bead_xyz = representative_xyz[bead.atom_indices]
        weights = np.array(bead.x_weights)[:, np.newaxis]
        com = np.sum(bead_xyz * weights, axis=0) / np.sum(weights)
        positions.append(com)
    return np.array(positions)


class CGViewer:
    """3Dmol.js viewer for CG beads with hover labels and bond highlighting."""

    BEAD_COLOR = '#4A90E2'
    HIGHLIGHT_BOND_COLOR = '#FFD700'  # Gold for highlighted bond

    def __init__(self, width=600, height=450):
        self.width = width
        self.height = height
        self._bead_positions = None
        self._bead_names = []
        self._bonds = []
        self._angles = []
        self._dihedrals = []
        self._show_angles = False
        self._show_dihedrals = False
        self._needs_zoom_reset = True
        self._highlighted_bond = None  # (i, j) or None

        self._html_pane = pn.pane.HTML(
            self._empty_html(),
            width=self.width,
            height=self.height + 20,
            sizing_mode='fixed',
        )

    def get_panel(self):
        """Return the Panel component."""
        return pn.Column(self._html_pane, sizing_mode='fixed',
                         height=self.height + 30)

    def set_beads(self, bead_positions: np.ndarray, bead_names: list, bonds=None,
                  angles=None, dihedrals=None):
        """Render CG beads."""
        self._bead_positions = np.array(bead_positions)
        self._bead_names = list(bead_names)
        self._bonds = bonds or []
        self._angles = angles or []
        self._dihedrals = dihedrals or []
        self._needs_zoom_reset = True
        self._update_html()

    def set_bonds(self, bonds):
        """Update bonds and refresh viewer."""
        self._bonds = bonds or []
        self._needs_zoom_reset = False
        self._update_html()

    def set_angles(self, angles, show=True):
        """Update angles for visualization."""
        self._angles = angles or []
        self._show_angles = show
        self._needs_zoom_reset = False
        self._update_html()

    def set_dihedrals(self, dihedrals, show=True):
        """Update dihedrals for visualization."""
        self._dihedrals = dihedrals or []
        self._show_dihedrals = show
        self._needs_zoom_reset = False
        self._update_html()

    def set_show_angles(self, show):
        """Toggle angle visualization."""
        self._show_angles = show
        self._needs_zoom_reset = False
        self._update_html()

    def set_show_dihedrals(self, show):
        """Toggle dihedral visualization."""
        self._show_dihedrals = show
        self._needs_zoom_reset = False
        self._update_html()

    def set_highlighted_bond(self, i, j):
        """Highlight a specific bond in the viewer."""
        self._highlighted_bond = (i, j)
        self._needs_zoom_reset = False
        self._update_html()

    def clear_highlighted_bond(self):
        """Clear bond highlighting."""
        self._highlighted_bond = None
        self._needs_zoom_reset = False
        self._update_html()

    def _empty_html(self):
        doc = f"""<!DOCTYPE html>
<html><head><style>body{{margin:0;overflow:hidden}}</style></head>
<body><div style="width:{self.width}px;height:{self.height}px;background:#1a1a2e;display:flex;align-items:center;justify-content:center;color:#ccc;font-family:sans-serif;">No beads to display</div></body></html>"""
        return f'<iframe srcdoc="{html.escape(doc, quote=True)}" style="width:{self.width}px;height:{self.height}px;border:none;"></iframe>'

    def _update_html(self):
        if self._bead_positions is None:
            self._html_pane.object = self._empty_html()
            return
        self._html_pane.object = self._build_html()

    def _is_highlighted_bond(self, i, j):
        if self._highlighted_bond is None:
            return False
        hi, hj = self._highlighted_bond
        return (i == hi and j == hj) or (i == hj and j == hi)

    def _build_html(self):
        n = len(self._bead_positions)
        lines = [f"{n}", "CG beads"]
        for i in range(n):
            x, y, z = self._bead_positions[i]
            lines.append(f"X {x:.6f} {y:.6f} {z:.6f}")
        xyz_str = "\\n".join(lines)

        # Build per-atom style specs
        style_specs = []
        for i in range(n):
            style_specs.append(
                f'viewer.setStyle({{index:{i}}}, {{sphere:{{radius:1.0,color:"{self.BEAD_COLOR}"}}}});'
            )
        styles_js = "\n".join(style_specs)

        # Build bond cylinders
        cylinder_specs = []
        for (i, j) in self._bonds:
            if i >= n or j >= n:
                continue
            xi, yi, zi = self._bead_positions[i]
            xj, yj, zj = self._bead_positions[j]
            if self._is_highlighted_bond(i, j):
                color = self.HIGHLIGHT_BOND_COLOR
                radius = 0.25
            else:
                color = "silver"
                radius = 0.15
            cylinder_specs.append(
                f'viewer.addCylinder({{start:{{x:{xi:.4f},y:{yi:.4f},z:{zi:.4f}}},end:{{x:{xj:.4f},y:{yj:.4f},z:{zj:.4f}}},radius:{radius},fromCap:1,toCap:1,color:"{color}"}});'
            )
        cylinders_js = "\n".join(cylinder_specs)

        # Build dihedral lines
        dihedral_specs = []
        if self._show_dihedrals:
            for (i, j, k, l) in self._dihedrals:
                if max(i, j, k, l) >= n:
                    continue
                xi, yi, zi = self._bead_positions[i]
                xl, yl, zl = self._bead_positions[l]
                dihedral_specs.append(
                    f'viewer.addLine({{start:{{x:{xi:.4f},y:{yi:.4f},z:{zi:.4f}}},end:{{x:{xl:.4f},y:{yl:.4f},z:{zl:.4f}}},dashed:true,linewidth:2,color:"orange"}});'
                )
        dihedrals_js = "\n".join(dihedral_specs)

        # Build names array for hover labels
        names_js = "[" + ",".join(f'"{name}"' for name in self._bead_names) + "]"

        viewer_id = "cg_viewer_container"

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
    var viewer = $3Dmol.createViewer(element, {{backgroundColor: "0x1a1a2e"}});
    var xyz = "{xyz_str}";
    var names = {names_js};
    viewer.addModel(xyz, "xyz");
    {styles_js}
    {cylinders_js}
    {dihedrals_js}

    // Hover labels: show bead name/index on hover
    viewer.setHoverable({{}}, true,
        function(atom, viewer, event, container) {{
            if (atom && atom.index !== undefined) {{
                var label = names[atom.index] + " (" + atom.index + ")";
                viewer.addLabel(label, {{
                    position: atom,
                    fontSize: 12,
                    backgroundColor: "rgba(0,0,0,0.7)",
                    fontColor: "white",
                    showBackground: true
                }});
                viewer.render();
            }}
        }},
        function(atom, viewer) {{
            viewer.removeAllLabels();
            viewer.render();
        }}
    );

    viewer.zoomTo();
    viewer.render();
}})();
</script>
</body>
</html>"""

        iframe = f'<iframe srcdoc="{html.escape(inner_doc, quote=True)}" style="width:{self.width}px;height:{self.height}px;border:none;"></iframe>'
        return iframe
