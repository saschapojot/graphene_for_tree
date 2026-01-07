import sys
import os
import pickle
import base64
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path
from matplotlib.patches import FancyArrowPatch, Circle
import matplotlib.lines as mlines

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from classes.class_defs import atomIndex, hopping, vertex

err_code_no_data = 10
err_read_exception = 11

# ==============================================================================
#  Read and Deserialize Data
# ==============================================================================
try:
    # Read the Base64 string from stdin
    raw_input = sys.stdin.read()
    if not raw_input:
        print("Error: No input data received via stdin.")
        sys.exit(err_code_no_data)
    # Decode Base64 -> Bytes -> Unpickle
    binary_data = base64.b64decode(raw_input)
    data_package = pickle.loads(binary_data)
    # Extract the objects
    all_roots_sorted = data_package["roots"]
    parsed_config = data_package["config"]
    unit_cell_atoms = data_package["unit_cell_atoms"]
    print(f"Successfully received {len(all_roots_sorted)} roots.")
    print(f"Configuration loaded for: {parsed_config.get('name', 'Unknown System')}")
except Exception as e:
    print(f"Error deserializing data: {e}")
    sys.exit(err_read_exception)


# ==============================================================================
#  Helper Functions
# ==============================================================================
def assign_line_types(root_vertex):
    """
    Traverses the constraint tree and assigns line_type to hoppings.
    Logic:
    - Root: line_type = 0
    - Linear Child: line_type = parent's line_type
    - Hermitian Child: line_type = reverted parent's line_type (0->1, 1->0)
    """

    def _traverse_recursive(node, current_line_type):
        # Assign the determined line_type to the current hopping
        node.hopping.line_type = current_line_type
        # Traverse children
        for child in node.children:
            next_line_type = current_line_type
            # If child is hermitian, flip the type (0->1, 1->0)
            if child.type == "hermitian":
                next_line_type = 1 - current_line_type
            # If child is linear, next_line_type remains current_line_type
            _traverse_recursive(child, next_line_type)

    # Start traversal. Root is always 0.
    if root_vertex.is_root:
        _traverse_recursive(root_vertex, 0)


def get_lattice_vectors_from_tree(root_vertex):
    """
    Traverses a single constraint tree starting from 'root_vertex' to extract
    [n0, n1, n2] vectors for the atoms in every hopping of every node.
    """
    extracted_vectors = []

    def _traverse_recursive(node):
        hop = node.hopping
        to_cell = [hop.to_atom.n0, hop.to_atom.n1, hop.to_atom.n2]
        from_cell = [hop.from_atom.n0, hop.from_atom.n1, hop.from_atom.n2]
        extracted_vectors.append(to_cell)
        extracted_vectors.append(from_cell)
        for child in node.children:
            _traverse_recursive(child)

    _traverse_recursive(root_vertex)
    return extracted_vectors

def get_extreme_vectors(lattice_set):
    """
    Finds the vectors containing the maximum and minimum n0 and n1.
    """
    if not lattice_set:
        return {
            "n0_max": {"value": 0, "vector": (0,0,0)},
            "n0_min": {"value": 0, "vector": (0,0,0)},
            "n1_max": {"value": 0, "vector": (0,0,0)},
            "n1_min": {"value": 0, "vector": (0,0,0)},
        }

    vec_max_n0 = max(lattice_set, key=lambda v: v[0])
    vec_min_n0 = min(lattice_set, key=lambda v: v[0])
    vec_max_n1 = max(lattice_set, key=lambda v: v[1])
    vec_min_n1 = min(lattice_set, key=lambda v: v[1])

    results = {
        "n0_max": {"value": vec_max_n0[0], "vector": vec_max_n0},
        "n0_min": {"value": vec_min_n0[0], "vector": vec_min_n0},
        "n1_max": {"value": vec_max_n1[1], "vector": vec_max_n1},
        "n1_min": {"value": vec_min_n1[1], "vector": vec_min_n1},
    }
    return results

def expand_vector_bounds(vector):
    """
    Adjusts a vector's elements: adds 1 if positive, subtracts 1 if negative.
    """
    adjusted = []
    for val in vector:
        if val > 0:
            adjusted.append(val + 1)
        elif val < 0:
            adjusted.append(val - 1)
        else:
            adjusted.append(val)
    return tuple(adjusted)


def get_real_coords(atom_idx_obj, a0, a1, a2):
    """
    Computes the real-space 3D coordinates (x, y, z) for an atomIndex object.
    """
    n0, n1 = atom_idx_obj.n0, atom_idx_obj.n1
    f0, f1, f2 = atom_idx_obj.frac_coord
    pos_vec = (n0 + f0) * a0 + (n1 + f1) * a1 + f2 * a2
    return pos_vec[0], pos_vec[1], pos_vec[2]


def draw_basis_vectors(ax, origin, a0, a1):
    """
    Draws the lattice basis vectors a0 and a1 starting from a specific origin point.
    """
    ox, oy = origin

    # Draw a0
    # shrinkA=0 and shrinkB=0 ensure the arrow starts exactly at 'origin'
    ax.add_patch(FancyArrowPatch(
        (ox, oy), (ox + a0[0], oy + a0[1]),
        arrowstyle='-|>', mutation_scale=15, color='black', linewidth=2,
        shrinkA=0, shrinkB=0, zorder=20
    ))
    ax.text(ox + a0[0], oy + a0[1], r"$\mathbf{a}_0$", fontsize=14, fontweight='bold', zorder=20)

    # Draw a1
    ax.add_patch(FancyArrowPatch(
        (ox, oy), (ox + a1[0], oy + a1[1]),
        arrowstyle='-|>', mutation_scale=15, color='black', linewidth=2,
        shrinkA=0, shrinkB=0, zorder=20
    ))
    ax.text(ox + a1[0], oy + a1[1], r"$\mathbf{a}_1$", fontsize=14, fontweight='bold', zorder=20)

def draw_self_hopping_loop(ax, atom_x, atom_y, atom_z, color, linestyle, radius=0.2):
    """
    Draws a circle with an arrow head on it to represent self-hopping.
    """
    offset_angle_deg = 45
    offset_angle_rad = np.deg2rad(offset_angle_deg)

    circle_center_x = atom_x + radius * np.cos(offset_angle_rad)
    circle_center_y = atom_y + radius * np.sin(offset_angle_rad)

    circle = Circle((circle_center_x, circle_center_y), radius,
                    color=color, fill=False, linestyle=linestyle, linewidth=1, zorder=12)
    ax.add_patch(circle)

    arrow_angle_rad = offset_angle_rad
    arrow_x = circle_center_x + radius * np.cos(arrow_angle_rad)
    arrow_y = circle_center_y + radius * np.sin(arrow_angle_rad)

    tangent_angle = arrow_angle_rad + np.pi / 2
    dx = np.cos(tangent_angle) * 0.001
    dy = np.sin(tangent_angle) * 0.001

    arrow = FancyArrowPatch((arrow_x, arrow_y), (arrow_x + dx, arrow_y + dy),
                            arrowstyle='-|>',
                            mutation_scale=10,
                            color=color,
                            zorder=12)
    ax.add_patch(arrow)


def draw_arrows_and_circles(root_vertex, ax, radius, a0, a1, a2, tolerance=1e-5):
    """
    Draws a circle around the root's 'to_atom' and arrows for all hoppings in the tree.
    """
    # Draw Truncation Circle around the Root's Center Atom
    center_atom = root_vertex.hopping.to_atom
    cx, cy, cz = get_real_coords(center_atom, a0, a1, a2)

    circle = Circle((cx, cy), radius, color='pink', fill=False,
                    linestyle='--', linewidth=1, zorder=8)
    ax.add_patch(circle)
    ax.scatter([cx], [cy], c='pink', s=5, zorder=15)

    def _traverse_draw(node):
        hop = node.hopping
        start_x, start_y, start_z = get_real_coords(hop.from_atom, a0, a1, a2)
        end_x, end_y, end_z = get_real_coords(hop.to_atom, a0, a1, a2)

        if getattr(hop, 'line_type', 0) == 1:
            arrow_color = 'blue'
            arrow_style = 'dotted'
        else:
            arrow_color = 'crimson'
            arrow_style = 'solid'

        # Check for self-hopping
        if abs(start_x - end_x) < tolerance and abs(start_y - end_y) < tolerance and abs(start_z - end_z) < tolerance:
            draw_self_hopping_loop(ax, start_x, start_y, start_z, arrow_color, arrow_style)
        else:
            arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                    arrowstyle='-|>',
                                    mutation_scale=10,
                                    color=arrow_color,
                                    linestyle=arrow_style,
                                    linewidth=1,
                                    zorder=12)
            ax.add_patch(arrow)

        for child in node.children:
            _traverse_draw(child)

    _traverse_draw(root_vertex)


# ==============================================================================
#  Main Plotting Function
# ==============================================================================
def plot_single_root_tree(root_vertex, root_index, parsed_config, unit_cell_atoms, output_dir, grid_params):
    """
    Generates and saves a plot for a single constraint tree root.
    Uses pre-calculated grid_params to ensure consistent scaling across all plots.
    """
    # Unpack grid parameters
    n0_range = grid_params['n0_range']
    n1_range = grid_params['n1_range']
    n0_min, n0_max = grid_params['n0_min'], grid_params['n0_max']
    n1_min, n1_max = grid_params['n1_min'], grid_params['n1_max']

    # Setup Lattice Basis
    lattice_basis = np.array(parsed_config["lattice_basis"])
    a0, a1, a2 = lattice_basis
    truncation_radius = parsed_config["truncation_radius"]

    # Create Figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw Grid Lines
    grid_segments = []

    # Vertical lines (vary n1)
    for n0_idx in n0_range:
        p_start = n0_idx * a0 + n1_min * a1
        p_end = n0_idx * a0 + n1_max * a1
        grid_segments.append([p_start[:2], p_end[:2]])

    # Horizontal lines (vary n0)
    for n1_idx in n1_range:
        p_start = n0_min * a0 + n1_idx * a1
        p_end = n0_max * a0 + n1_idx * a1
        grid_segments.append([p_start[:2], p_end[:2]])

    lc = LineCollection(grid_segments, colors='grey', linewidths=1.0, alpha=0.6)
    ax.add_collection(lc)

    # Highlight Unit Cell [0,0]
    c0 = 0 * a0 + 0 * a1
    c1 = 1 * a0 + 0 * a1
    c2 = 1 * a0 + 1 * a1
    c3 = 0 * a0 + 1 * a1
    highlight_segments = [
        [c0[:2], c1[:2]], [c1[:2], c2[:2]],
        [c2[:2], c3[:2]], [c3[:2], c0[:2]]
    ]
    lc_highlight = LineCollection(highlight_segments, colors='black',
                                  linewidths=2.5, alpha=1.0, zorder=5)
    ax.add_collection(lc_highlight)

    # Draw Atoms
    unique_position_names = set(atom.position_name for atom in unit_cell_atoms)
    sorted_position_names = sorted(list(unique_position_names))
    num_unique_positions = len(sorted_position_names)
    hsv_colors = plt.cm.hsv(np.linspace(0, 1, num_unique_positions, endpoint=False))
    name_to_color = {name: hsv_colors[i] for i, name in enumerate(sorted_position_names)}

    xs, ys, plot_colors = [], [], []

    # Populate atoms for the calculated grid range
    for n0 in n0_range[:-1]:
        for n1 in n1_range[:-1]:
            for atom in unit_cell_atoms:
                f0, f1, f2 = atom.frac_coord
                pos = (n0 + f0) * a0 + (n1 + f1) * a1 + f2 * a2
                xs.append(pos[0])
                ys.append(pos[1])
                plot_colors.append(name_to_color[atom.position_name])

    ax.scatter(xs, ys, c=plot_colors, s=40, edgecolors='black', zorder=10)

    # Add Legend
    legend_elements = []
    for name, color in name_to_color.items():
        legend_elements.append(mlines.Line2D([], [], color=color, marker='o',
                                             linestyle='None', markersize=10, label=name,
                                             markeredgecolor='black'))
    ax.legend(handles=legend_elements, loc='upper right', title="Wyckoff Pos")

    # Draw Tree Arrows
    draw_arrows_and_circles(root_vertex, ax, truncation_radius, a0, a1, a2)

    # Set Limits and Labels
    corner1 = n0_min * a0 + n1_min * a1
    corner2 = n0_max * a0 + n1_min * a1
    corner3 = n0_max * a0 + n1_max * a1
    corner4 = n0_min * a0 + n1_max * a1
    all_x = [corner1[0], corner2[0], corner3[0], corner4[0]]
    all_y = [corner1[1], corner2[1], corner3[1], corner4[1]]

    padding = 0.5
    min_x_val = min(all_x) - padding
    max_x_val = max(all_x) + padding
    min_y_val = min(all_y) - padding
    max_y_val = max(all_y) + padding

    ax.set_xlim(min_x_val, max_x_val)
    ax.set_ylim(min_y_val, max_y_val)
    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # Draw Basis Vectors in Bottom Left Corner
    # Place origin slightly offset from the absolute corner of the plot limits
    basis_origin = (corner1[0], corner1[1])
    draw_basis_vectors(ax, basis_origin, a0, a1)

    hop = root_vertex.hopping
    title_str = f"Tree {root_index}: {hop.to_atom.position_name} <- {hop.from_atom.position_name} (d={hop.distance:.4f})"
    ax.set_title(title_str)

    # Save Plot
    filename = f"lattice_grid_tree_{root_index}.png"
    output_file = os.path.join(output_dir, filename)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close(fig)

# ==============================================================================
#  Main Execution
# ==============================================================================

# 1. Pre-process all trees to assign line types
for root in all_roots_sorted:
    assign_line_types(root)


# 2. CALCULATE GLOBAL GRID RANGES (Common for all plots)
# Collect all lattice vectors from ALL trees
lattices_all = set()
for one_root in all_roots_sorted:
    extracted_vectors = get_lattice_vectors_from_tree(one_root)
    for vec in extracted_vectors:
        lattices_all.add(tuple(vec))

# Get extreme vectors
targets = get_extreme_vectors(lattices_all)

print("-" * 50)
print(f"Largest positive n0:  {targets['n0_max']['value']} | Vector: {targets['n0_max']['vector']}")
print(f"Smallest negative n0: {targets['n0_min']['value']} | Vector: {targets['n0_min']['vector']}")
print("-" * 50)
print(f"Largest positive n1:  {targets['n1_max']['value']} | Vector: {targets['n1_max']['vector']}")
print(f"Smallest negative n1: {targets['n1_min']['value']} | Vector: {targets['n1_min']['vector']}")
print("-" * 50)

# Expand bounds
vec_n0_max_expanded = expand_vector_bounds(targets['n0_max']['vector'])
vec_n0_min_expanded = expand_vector_bounds(targets['n0_min']['vector'])
vec_n1_max_expanded = expand_vector_bounds(targets['n1_max']['vector'])
vec_n1_min_expanded = expand_vector_bounds(targets['n1_min']['vector'])


# Define ranges
max_n0_val = vec_n0_max_expanded[0]
min_n0_val = vec_n0_min_expanded[0]
max_n1_val = vec_n1_max_expanded[1]
min_n1_val = vec_n1_min_expanded[1]

# Package grid params to pass to function
grid_params = {
    'n0_range': list(range(min_n0_val, max_n0_val + 1)),
    'n1_range': list(range(min_n1_val, max_n1_val + 1)),
    'n0_min': min_n0_val,
    'n0_max': max_n0_val,
    'n1_min': min_n1_val,
    'n1_max': max_n1_val
}
print("\n" + "=" * 50)
print("GLOBAL GRID GENERATION RANGES")
print("=" * 50)
print(f"n0 Range ({min_n0_val} to {max_n0_val}): {grid_params['n0_range']}")
print(f"n1 Range ({min_n1_val} to {max_n1_val}): {grid_params['n1_range']}")
print("=" * 50)

# 3. Setup Output Directory
config_file_path = parsed_config["config_file_path"]
config_dir = Path(config_file_path).parent
output_dir = str(config_dir) + "/tree_visualization/"
Path(output_dir).mkdir(parents=True, exist_ok=True)

# 4. Iterate and Plot
print(f"\nGenerating plots for {len(all_roots_sorted)} trees...")
for i, root in enumerate(all_roots_sorted):
    # Only plot if it is a root (though input should be roots)
    if root.is_root:
        plot_single_root_tree(root, i, parsed_config, unit_cell_atoms, output_dir, grid_params)
    else:
        print(f"Skipping index {i} as it is not a root vertex.")

print("\nAll plots generated.")