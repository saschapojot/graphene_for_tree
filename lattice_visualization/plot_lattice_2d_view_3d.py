import sys
import os
import pickle
import base64
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


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
    n0, n1,n2 = atom_idx_obj.n0, atom_idx_obj.n1, atom_idx_obj.n2
    f0, f1, f2 = atom_idx_obj.frac_coord
    pos_vec = (n0 + f0) * a0 + (n1 + f1) * a1 +(n2+ f2) * a2
    return pos_vec[0], pos_vec[1], pos_vec[2]


def plot_single_root_tree(root_vertex, root_index, parsed_config, unit_cell_atoms, output_dir, grid_params):
    # Unpack grid parameters
    n0_range = grid_params['n0_range']
    n1_range = grid_params['n1_range']
    n0_min, n0_max = grid_params['n0_min'], grid_params['n0_max']
    n1_min, n1_max = grid_params['n1_min'], grid_params['n1_max']

    # Setup Lattice Basis
    lattice_basis = np.array(parsed_config["lattice_basis"])
    a0, a1, a2 = lattice_basis
    truncation_radius = parsed_config["truncation_radius"]
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')


    # Enable grid FIRST
    ax.grid(True)
    # THEN make grid lines on all planes very dim (increase alpha to be more visible)
    ax.xaxis._axinfo["grid"].update({"color": (0.5, 0.5, 0.5, 0.2), "linewidth": 0.5})
    ax.yaxis._axinfo["grid"].update({"color": (0.5, 0.5, 0.5, 0.2), "linewidth": 0.5})
    ax.zaxis._axinfo["grid"].update({"color": (0.5, 0.5, 0.5, 0.2), "linewidth": 0.5})

    #plot lattice lines along a0 direction
    for n1 in range(n1_min,n1_max+1):
        starting_point=n1*a1+n0_min*a0
        ending_point=n1*a1+n0_max*a0
        x_start,y_start,z_start=starting_point
        x_end,y_end,z_end=ending_point

        x_vec=[x_start,x_end]
        y_vec=[y_start,y_end]
        z_vec=[z_start,z_end]

        ax.plot3D(x_vec, y_vec, z_vec, color='black', linewidth=1, linestyle='-')
    #plot lattice along a1 direction
    for n0 in range(n0_min,n0_max+1):
        starting_point=n0*a0+n1_min*a1
        ending_point=n0*a0+n1_max*a1
        x_start, y_start, z_start = starting_point
        x_end, y_end, z_end = ending_point

        x_vec = [x_start, x_end]
        y_vec = [y_start, y_end]
        z_vec = [z_start, z_end]
        ax.plot3D(x_vec, y_vec, z_vec, color='black', linewidth=1, linestyle='-')

    # Draw Atoms
    unique_position_names = set(atom.position_name for atom in unit_cell_atoms)
    sorted_position_names = sorted(list(unique_position_names))
    num_unique_positions = len(sorted_position_names)
    hsv_colors = plt.cm.hsv(np.linspace(0, 1, num_unique_positions, endpoint=False))
    name_to_color = {name: hsv_colors[i] for i, name in enumerate(sorted_position_names)}


    # Plot each atom type separately to create legend entries
    for position_name in sorted_position_names:
        x_atoms, y_atoms, z_atoms = [], [], []
        # Populate atoms for this specific atom type
        for n0 in n0_range[:-1]:
            for n1 in n1_range[:-1]:
                for atom in unit_cell_atoms:
                    if atom.position_name == position_name:
                        f0, f1, f2 = atom.frac_coord
                        pos = (n0 + f0) * a0 + (n1 + f1) * a1 + f2 * a2
                        x_atoms.append(pos[0])
                        y_atoms.append(pos[1])
                        z_atoms.append(pos[2])

        # Plot this atom type with its label
        ax.scatter3D(x_atoms, y_atoms, z_atoms,
                     c=[name_to_color[position_name]],
                     s=40,
                     edgecolors='black',
                     zorder=10,
                     label=position_name)  # This creates the legend entry


    ax.view_init(elev=30, azim=45)  # Adjust to your preference

    # Set labels
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('z', fontsize=12)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # ax.set_zlim(-10, 10)  # Set z-axis limits

    # Save Plot
    filename = f"lattice_grid_tree_{root_index}.png"
    output_file = os.path.join(output_dir, filename)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    # plt.show()
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
output_dir = str(config_dir) + "/tree_visualization_3d_view/"
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














