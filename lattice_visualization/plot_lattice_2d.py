import sys
import os
import pickle
import base64
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from classes.class_defs import atomIndex,hopping,vertex

err_code_no_data=10
err_read_exception=11
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
    print(f"Successfully received {len(all_roots_sorted)} roots.")
    print(f"Configuration loaded for: {parsed_config.get('name', 'Unknown System')}")
except Exception as e:
    print(f"Error deserializing data: {e}")
    sys.exit(err_read_exception)


def print_tree(root, prefix="", is_last=True, show_details=True, max_depth=None, current_depth=0):
    """
    Print a constraint tree structure in a visual hierarchical format.

    Args:
        root: vertex object (root of tree or subtree)
        prefix: String prefix for indentation (used in recursion)
        is_last: Boolean indicating if this is the last child (affects connector style)
        show_details: Whether to show detailed hopping information (default: True)
        max_depth: Maximum depth to print (None = unlimited, default: None)
        current_depth: Current depth in recursion (internal use, default: 0)

    Tree Structure Symbols:
        ╔═══ ROOT     (root node)
        ├── CHILD    (middle child)
        └── CHILD    (last child)
        │           (vertical line for continuation)

    Example Output:
        ╔═══ ROOT: N[0,0,0] ← N[0,0,0], op=0, d=0.0000
        ├── CHILD (linear): N[0,0,0] ← N[1,0,0], op=1, d=2.5000
        ├── CHILD (linear): N[0,0,0] ← N[-1,1,0], op=2, d=2.5000
        └── CHILD (linear): N[0,0,0] ← N[0,-1,0], op=3, d=2.5000
    """
    # Check max depth
    if max_depth is not None and current_depth > max_depth:
        return

    # Determine node styling
    if root.is_root:
        node_label = "ROOT"
        connector = "╔═══ "
        detail_prefix = prefix
    else:
        node_label = f"CHILD ({root.type})"
        connector = "└── " if is_last else "├── "
        detail_prefix = prefix + ("    " if is_last else "│   ")

    # Build node description
    hop = root.hopping

    # Basic info: atom types and operation
    to_cell = f"[{hop.to_atom.n0},{hop.to_atom.n1},{hop.to_atom.n2}]"
    from_cell = f"[{hop.from_atom.n0},{hop.from_atom.n1},{hop.from_atom.n2}]"
    basic_info = f"{hop.to_atom.wyckoff_instance_id}{to_cell} ← {hop.from_atom.wyckoff_instance_id}{from_cell}"

    # Print main node line
    if show_details:
        print(f"{prefix}{connector}{node_label}: {basic_info}, "
              f"op={hop.operation_idx}, dist={hop.distance:.4f}")
    else:
        print(f"{prefix}{connector}{node_label}: op={hop.operation_idx}")

    # Print additional details if requested and this is root
    if show_details and root.is_root and current_depth == 0:
        print(f"{detail_prefix}    ├─ Type: {root.type}")
        print(f"{detail_prefix}    ├─ Children: {len(root.children)}")
        print(f"{detail_prefix}    └─ Distance: {hop.distance:.6f}")

    # Recursively print children
    if root.children:
        for i, child in enumerate(root.children):
            is_last_child = (i == len(root.children) - 1)

            # Determine new prefix for children
            if root.is_root:
                new_prefix = ""
            else:
                new_prefix = prefix + ("    " if is_last else "│   ")

            print_tree(child, new_prefix, is_last_child, show_details, max_depth, current_depth + 1)

def print_all_trees(roots_list, show_details=True, max_trees=None, max_depth=None):
    """
    Print all constraint trees in a formatted way.

    Args:
        roots_list: List of root vertex objects
        show_details: Whether to show detailed information (default: True)
        max_trees: Maximum number of trees to print (None = all, default: None)
        max_depth: Maximum depth to print for each tree (None = unlimited, default: None)
    """
    print("\n" + "=" * 80)
    print("CONSTRAINT TREE STRUCTURES")
    print("=" * 80)

    # CRITICAL FIX: Filter to only include actual roots (is_root == True)
    # ================================================================
    # ADD THIS LINE RIGHT HERE - it filters out grafted vertices
    actual_roots = [root for root in roots_list if root.is_root]

    # Print diagnostic if non-root vertices found in the list
    if len(actual_roots) < len(roots_list):
        print(f"\nNote: Input list contained {len(roots_list)} vertices")
        print(f"      Filtered to {len(actual_roots)} actual roots")
        print(f"      ({len(roots_list) - len(actual_roots)} vertices were grafted as hermitian children)\n")

    # Use actual_roots instead of roots_list for counting
    num_trees = len(actual_roots) if max_trees is None else min(max_trees, len(actual_roots))

    for i in range(num_trees):
        root = actual_roots[i]  # Changed from roots_list[i] to actual_roots[i]
        hop = root.hopping

        print(f"\n{'─' * 80}")
        print(f"Tree {i}: Distance = {hop.distance:.6f}, "
              f"Hopping: {hop.to_atom.position_name} ← {hop.from_atom.position_name}")
        print(f"{'─' * 80}")

        print_tree(root, show_details=show_details, max_depth=max_depth)

    if max_trees is not None and len(actual_roots) > max_trees:
        print(f"\n... and {len(actual_roots) - max_trees} more trees")

    print("\n" + "=" * 80)


def get_lattice_vectors_from_tree(root_vertex):
    """
     Traverses a single constraint tree starting from 'root_vertex' to extract
     [n0, n1, n2] vectors for the atoms in every hopping of every node.
    Args:
        root_vertex:  The root vertex object of the tree to traverse.

    Returns:

    """
    extracted_vectors  = []

    def _traverse_recursive(node):
        """
        Helper function to recursively visit children.
        Args:
            node:
            depth:

        Returns:

        """
        # Access the hopping object within the current vertex
        hop = node.hopping
        # 1. Extract [n0, n1, n2] for the 'To Atom' (Center/Destination)
        # These integers are stored directly in the atomIndex object
        to_cell = [hop.to_atom.n0, hop.to_atom.n1, hop.to_atom.n2]
        # 2. Extract [n0, n1, n2] for the 'From Atom' (Neighbor/Source)
        from_cell = [hop.from_atom.n0, hop.from_atom.n1, hop.from_atom.n2]

        extracted_vectors .append(to_cell)
        extracted_vectors .append(from_cell)
        # Recurse through children
        for child in node.children:
            _traverse_recursive(child)
    # Start recursion from the provided root
    _traverse_recursive(root_vertex)
    return extracted_vectors

def get_extreme_vectors(lattice_set):
    """
    Finds the vectors containing the maximum and minimum n0 and n1.
    Args:
        lattice_set: A set or list of tuples (n0, n1, n2)

    Returns:
        results: A dictionary containing the value and the full vector.
    """
    # Find vector with the largest n0 (index 0)
    vec_max_n0 = max(lattice_set, key=lambda v: v[0])
    # Find vector with the smallest n0 (index 0)
    vec_min_n0 = min(lattice_set, key=lambda v: v[0])
    # Find vector with the largest n1 (index 1)
    vec_max_n1 = max(lattice_set, key=lambda v: v[1])

    # Find vector with the smallest n1 (index 1)
    vec_min_n1 = min(lattice_set, key=lambda v: v[1])

    results = {
        "n0_max": {"value": vec_max_n0[0], "vector": vec_max_n0},
        "n0_min": {"value": vec_min_n0[0], "vector": vec_min_n0},
        "n1_max": {"value": vec_max_n1[1], "vector": vec_max_n1},
        "n1_min": {"value": vec_min_n1[1], "vector": vec_min_n1},
    }
    return results

lattices_all=set()
for one_root in all_roots_sorted:
    extracted_vectors=get_lattice_vectors_from_tree(one_root)
    # Iterate over the extracted vectors
    for vec in extracted_vectors:
        # Convert the list [n0, n1, n2] to a tuple (n0, n1, n2)
        # before adding it to the set, as lists are not hashable.
        lattices_all.add(tuple(vec))


def expand_vector_bounds(vector):
    """
    Adjusts a vector's elements: adds 1 if positive, subtracts 1 if negative,
    leaves 0 as is.

    Args:
        vector: A tuple or list (n0, n1, n2)

    Returns:
        tuple: The adjusted vector (n0', n1', n2')
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

targets = get_extreme_vectors(lattices_all)
print("-" * 50)
print(f"Largest positive n0:  {targets['n0_max']['value']} | Vector: {targets['n0_max']['vector']}")
print(f"Smallest negative n0: {targets['n0_min']['value']} | Vector: {targets['n0_min']['vector']}")
print("-" * 50)
print(f"Largest positive n1:  {targets['n1_max']['value']} | Vector: {targets['n1_max']['vector']}")
print(f"Smallest negative n1: {targets['n1_min']['value']} | Vector: {targets['n1_min']['vector']}")
print("-" * 50)

# ==============================================================================
#  Calculate Grid Ranges
# ==============================================================================

# Expand the extreme vectors to ensure we cover the boundary
vec_n0_max_expanded = expand_vector_bounds(targets['n0_max']['vector'])
vec_n0_min_expanded = expand_vector_bounds(targets['n0_min']['vector'])
vec_n1_max_expanded = expand_vector_bounds(targets['n1_max']['vector'])
vec_n1_min_expanded = expand_vector_bounds(targets['n1_min']['vector'])


# Extract the specific coordinate values from the expanded vectors
#    n0 is index 0, n1 is index 1
max_n0_val = vec_n0_max_expanded[0]
min_n0_val = vec_n0_min_expanded[0]
max_n1_val = vec_n1_max_expanded[1]
min_n1_val = vec_n1_min_expanded[1]

# Generate the grid values (inclusive ranges)
#    We use range(start, stop + 1) to include the upper bound
n0_range = list(range(min_n0_val, max_n0_val + 1))
n1_range = list(range(min_n1_val, max_n1_val + 1))

print("\n" + "=" * 50)
print("GRID GENERATION RANGES")
print("=" * 50)
print(f"n0 Range ({min_n0_val} to {max_n0_val}): {n0_range}")
print(f"n1 Range ({min_n1_val} to {max_n1_val}): {n1_range}")
print("=" * 50)

print(parsed_config)
lattice_basis=np.array(parsed_config["lattice_basis"])
a0,a1,_=lattice_basis

config_file_path=parsed_config["config_file_path"]
# Get directory of config file and create tree_visualization folder
config_dir = Path(config_file_path).parent
output_dir = str(config_dir)+"/tree_visualization/"
Path(output_dir).mkdir(parents=True, exist_ok=True)

# ==============================================================================
#  Plot Lattice Cells
# ==============================================================================
# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 12))
grid_segments = []
# Determine min and max for ranges to define the grid boundaries
n0_min, n0_max = min(n0_range), max(n0_range)
n1_min, n1_max = min(n1_range), max(n1_range)
# Create segments parallel to a1 (varying n1, fixed n0)
# We iterate over every n0 in the range to draw the "vertical" lines
for n0_idx in n0_range:
    # Start point of line: n0*a0 + min_n1*a1
    p_start = n0_idx * a0 + n1_min * a1
    # End point of line: n0*a0 + max_n1*a1
    p_end = n0_idx * a0 + n1_max * a1
    # Append segment (using only x, y coordinates)
    grid_segments.append([p_start[:2], p_end[:2]])


# Create segments parallel to a0 (varying n0, fixed n1)
# We iterate over every n1 in the range to draw the "horizontal" lines
for n1_idx in n1_range:
    # Start point of line: min_n0*a0 + n1*a1
    p_start = n0_min * a0 + n1_idx * a1
    # End point of line: max_n0*a0 + n1*a1
    p_end = n0_max * a0 + n1_idx * a1
    # Append segment (using only x, y coordinates)
    grid_segments.append([p_start[:2], p_end[:2]])


# Create LineCollection for efficient plotting
lc = LineCollection(grid_segments, colors='grey', linewidths=1.0, alpha=0.6)
ax.add_collection(lc)
# Calculate limits to ensure the whole grid is visible
# We calculate the four corners of the entire grid patch
corner1 = n0_min * a0 + n1_min * a1
corner2 = n0_max * a0 + n1_min * a1
corner3 = n0_max * a0 + n1_max * a1
corner4 = n0_min * a0 + n1_max * a1

all_x = [corner1[0], corner2[0], corner3[0], corner4[0]]
all_y = [corner1[1], corner2[1], corner3[1], corner4[1]]
# Add some padding
padding = 2.0
ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
ax.set_ylim(min(all_y) - padding, max(all_y) + padding)

# Ensure aspect ratio is equal so lattice vectors aren't distorted
ax.set_aspect('equal')
# Add labels and title
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"Lattice Grid for {parsed_config.get('name', 'System')}")
# Save the plot
output_filename = "lattice_grid.png"
save_path = os.path.join(output_dir, output_filename)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Lattice grid plot saved to: {save_path}")
