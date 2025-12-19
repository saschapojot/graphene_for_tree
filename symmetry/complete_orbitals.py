import numpy as np
import sys
import re
import json
import copy

# ==============================================================================
# Orbital completeness checker and symmetry-based orbital completion script
# ==============================================================================
# Original file: /home/adada/Documents/pyCode/TB/ck/CheckOrbCpl.py
#
# This script checks if the orbitals specified by the user form a complete set
# under space group symmetry operations. If orbitals are incomplete, it adds
# orbitals that are related by symmetry to ensure the tight-binding basis is
# closed under the symmetry group.
#
# Example: If user specifies only 2pz for an atom, but symmetry couples it to
# 2px and 2py, the script will automatically add 2px and 2py to the basis.

# Exit codes
json_err_code = 4   # JSON parsing error


# ==============================================================================
# STEP 1: Read and parse JSON input from stdin
# ==============================================================================
try:
    combined_input_json = sys.stdin.read()
    combined_input = json.loads(combined_input_json)
except json.JSONDecodeError as e:
    print(f"Error parsing JSON input: {e}", file=sys.stderr)
    exit(json_err_code)


# ==============================================================================
# STEP 2: Define orbital indexing system
# ==============================================================================
# Global mapping from orbital names to indices (0-77)
# This covers all orbitals from 1s to 7f
# Total: 78 orbitals organized by principal quantum number and angular momentum
orbital_map = {
    # n=1: 1s (index 0)
    '1s': 0,

    # n=2: 2s, 2p (indices 1-4)
    '2s': 1,
    '2px': 2, '2py': 3, '2pz': 4,

    # n=3: 3s, 3p, 3d (indices 5-13)
    '3s': 5,
    '3px': 6, '3py': 7, '3pz': 8,
    '3dxy': 9, '3dyz': 10, '3dxz': 11, '3dx2-y2': 12, '3dz2': 13,

    # n=4: 4s, 4p, 4d, 4f (indices 14-29)
    '4s': 14,
    '4px': 15, '4py': 16, '4pz': 17,
    '4dxy': 18, '4dyz': 19, '4dxz': 20, '4dx2-y2': 21, '4dz2': 22,
    '4fxyz': 23, '4fz3': 24, '4fxz2': 25, '4fyz2': 26,
    '4fz(x2-y2)': 27, '4fx(x2-3y2)': 28, '4fy(3x2-y2)': 29,

    # n=5: 5s, 5p, 5d, 5f (indices 30-45)
    '5s': 30,
    '5px': 31, '5py': 32, '5pz': 33,
    '5dxy': 34, '5dyz': 35, '5dxz': 36, '5dx2-y2': 37, '5dz2': 38,
    '5fxyz': 39, '5fz3': 40, '5fxz2': 41, '5fyz2': 42,
    '5fz(x2-y2)': 43, '5fx(x2-3y2)': 44, '5fy(3x2-y2)': 45,

    # n=6: 6s, 6p, 6d, 6f (indices 46-61)
    '6s': 46,
    '6px': 47, '6py': 48, '6pz': 49,
    '6dxy': 50, '6dyz': 51, '6dxz': 52, '6dx2-y2': 53, '6dz2': 54,
    '6fxyz': 55, '6fz3': 56, '6fxz2': 57, '6fyz2': 58,
    '6fz(x2-y2)': 59, '6fx(x2-3y2)': 60, '6fy(3x2-y2)': 61,

    # n=7: 7s, 7p, 7d, 7f (indices 62-77)
    '7s': 62,
    '7px': 63, '7py': 64, '7pz': 65,
    '7dxy': 66, '7dyz': 67, '7dxz': 68, '7dx2-y2': 69, '7dz2': 70,
    '7fxyz': 71, '7fz3': 72, '7fxz2': 73, '7fyz2': 74,
    '7fz(x2-y2)': 75, '7fx(x2-3y2)': 76, '7fy(3x2-y2)': 77,
}


# ==============================================================================
# STEP 3: Extract configuration and space group data
# ==============================================================================
# Split the combined input into two main components
parsed_config = combined_input["parsed_config"]
space_group_representations = combined_input["space_group_representations"]

# Extract space group matrices in different coordinate systems
space_group_matrices = np.array(space_group_representations["space_group_matrices"])
space_group_matrices_cartesian = np.array(space_group_representations["space_group_matrices_cartesian"])
space_group_matrices_primitive = np.array(space_group_representations["space_group_matrices_primitive"])

# Extract representation matrices for different orbital angular momenta
# These show how symmetry operations transform s, p, d, f orbitals
repr_s, repr_p, repr_d, repr_f = space_group_representations["repr_s_p_d_f"]

repr_s_np = np.array(repr_s)
repr_p_np = np.array(repr_p)
repr_d_np = np.array(repr_d)
repr_f_np = np.array(repr_f)

# Get number of symmetry operations (N in notes)
num_operations, _, _ = repr_s_np.shape


# ==============================================================================
# STEP 4: Build combined orbital representation matrix
# ==============================================================================
# IndSPDF: Array defining the dimensionality of each orbital shell
# Structure: [1s, 2s, 2p, 3s, 3p, 3d, 4s, 4p, 4d, 4f, ...]
# Values: dimension of each shell (s=1, p=3, d=5, f=7)
orbital_nums_spdf = np.array([
    1,           # 1s
    1, 3,        # 2s, 2p
    1, 3, 5,     # 3s, 3p, 3d
    1, 3, 5, 7,  # 4s, 4p, 4d, 4f
    1, 3, 5, 7,  # 5s, 5p, 5d, 5f
    1, 3, 5, 7,  # 6s, 6p, 6d, 6f
    1, 3, 5, 7,  # 7s, 7p, 7d, 7f
])

print(f"len(orbital_nums_spdf)={len(orbital_nums_spdf)}", file=sys.stderr)

# Total dimension of orbital space (should be 78)
orbital_max_dim = np.sum(orbital_nums_spdf)
print(f"orbital_max_dim={orbital_max_dim}", file=sys.stderr)

# SymSPDF: Combined representation matrix for all orbitals
# Shape: (num_operations, 78, 78)
# This is a block diagonal matrix with blocks for each orbital shell
spdf_combined = np.zeros((num_operations, orbital_max_dim, orbital_max_dim))


# ==============================================================================
# STEP 5: Define function to build orbital vectors for each atom
# ==============================================================================
def build_orbital_vectors(parsed_config):
    """
    Build a length-78 orbital vector for each atom in the configuration

    The orbital vector is a binary array where 1 indicates an active orbital
    and 0 indicates an inactive orbital. This represents which orbitals are
    included in the tight-binding basis for each atom.

    Example:
    If atom B has orbitals ['2pz', '2s'], the vector will have 1's at
    indices corresponding to 2s and 2pz, and 0's elsewhere.

    :param parsed_config: Dictionary containing atom types and their orbitals
    :return: Dictionary mapping atom position names to their orbital vectors (78-dim binary arrays)
    """
    # Build vectors for each atom position
    atom_orbital_vectors = {}

    # Iterate over Wyckoff positions (formerly atom_positions)
    for atom in parsed_config['Wyckoff_positions']:
        position_name = atom['position_name']
        atom_type = atom['atom_type']

        # Get orbitals for this atom type from configuration
        # Using Wyckoff_position_types instead of atom_types
        orbitals = parsed_config['Wyckoff_position_types'][atom_type]['orbitals']

        # Create 78-dimensional binary orbital vector (all zeros initially)
        orbital_vector = np.zeros(78)

        # Set 1 for each active orbital
        for orbital in orbitals:
            if orbital in orbital_map:
                orbital_vector[orbital_map[orbital]] = 1
            else:
                print(f"Warning: Orbital '{orbital}' for atom '{position_name}' not recognized", file=sys.stderr)

        atom_orbital_vectors[position_name] = orbital_vector

    return atom_orbital_vectors


# ==============================================================================
# STEP 6: Fill the combined representation matrix with orbital blocks
# ==============================================================================
# Fill the diagonal blocks of spdf_combined
# Each block corresponds to one shell (e.g., 2s, 2p, 3d, etc.)
for j in range(num_operations):
    current_idx = 0

    # Iterate through orbital_nums_spdf to place each block
    for i, block_size in enumerate(orbital_nums_spdf):
        if block_size == 1:  # s orbital (1x1 block)
            spdf_combined[j, current_idx:current_idx+1, current_idx:current_idx+1] = repr_s_np[j]
        elif block_size == 3:  # p orbital (3x3 block)
            spdf_combined[j, current_idx:current_idx+3, current_idx:current_idx+3] = repr_p_np[j]
        elif block_size == 5:  # d orbital (5x5 block)
            spdf_combined[j, current_idx:current_idx+5, current_idx:current_idx+5] = repr_d_np[j]
        elif block_size == 7:  # f orbital (7x7 block)
            spdf_combined[j, current_idx:current_idx+7, current_idx:current_idx+7] = repr_f_np[j]

        current_idx += block_size


# ==============================================================================
# STEP 7: Find which orbitals are coupled by symmetry
# ==============================================================================
# IndNonZero: Boolean matrix indicating which orbitals are coupled by symmetry
# If non_zero_spdf_combined[i,j] is True, orbitals i and j are coupled by at least one symmetry operation
# This is computed by summing absolute values across all symmetry operations
non_zero_spdf_combined = np.sum(np.abs(spdf_combined), axis=0) > 1e-6


# ==============================================================================
# STEP 8: Build initial orbital vectors from user input
# ==============================================================================
# Create orbital vectors for each atom based on user-specified orbitals
atom_orbital_vectors = build_orbital_vectors(parsed_config)  # Binary vectors with 1 for active orbitals


# ==============================================================================
# STEP 9: Complete orbital sets using symmetry coupling
# ==============================================================================
# Update atom_orbital_vectors based on symmetry coupling
# If user specifies orbital A, and symmetry couples A to B, then B must also be included
updated_atom_orbital_vectors = {}
added_orbitals_dict = {}  # Dictionary to store which orbitals were added for each atom

for atom_name, orbital_vector in atom_orbital_vectors.items():
    # Start with a copy of the original vector
    updated_vector = copy.deepcopy(orbital_vector)

    # Find indices where the orbital vector has 1 (active orbitals)
    active_orbital_indices = np.where(orbital_vector == 1)[0]

    # For each active orbital
    for orbital_idx in active_orbital_indices:
        # Find all orbitals coupled to this one by symmetry
        # Look at column orbital_idx in non_zero_spdf_combined
        coupled_orbital_indices = np.where(non_zero_spdf_combined[:, orbital_idx])[0]

        # Set all coupled positions to 1 (add them to the basis)
        updated_vector[coupled_orbital_indices] = 1

    updated_atom_orbital_vectors[atom_name] = updated_vector

    # Report which orbitals were added
    added_indices = np.where((updated_vector == 1) & (orbital_vector == 0))[0]
    if len(added_indices) > 0:
        # Get orbital names for the added indices
        added_orbitals = [k for k, v in orbital_map.items() if v in added_indices]
        added_orbitals_dict[atom_name] = added_orbitals
    else:
        added_orbitals_dict[atom_name] = []  # Empty list if no orbitals added

# Replace the original vectors with updated (completed) ones
atom_orbital_vectors = updated_atom_orbital_vectors  # Now labels all symmetry-required orbitals with 1

print(f"atom_orbital_vectors={atom_orbital_vectors}", file=sys.stderr)
print(f"added_orbitals_dict={added_orbitals_dict}", file=sys.stderr)


# ==============================================================================
# STEP 10: Extract symmetry representations for active orbitals only
# ==============================================================================
# For each atom, extract the submatrices of symmetry representations
# that act only on its active orbitals (reduces dimension from 78x78 to n×n)
repr_on_active_orbitals = {}

for atom_name, orbital_vector in atom_orbital_vectors.items():
    # Find indices of active orbitals for this atom
    active_indices = np.where(orbital_vector == 1)[0]

    if len(active_indices) > 0:
        # Create index arrays for extracting submatrices
        # idx will select rows and columns corresponding to active orbitals
        idx = np.ix_(active_indices, active_indices)

        # Extract the symmetry matrices for just these orbitals
        repr_matrices_for_atom = []
        for sym_op in range(num_operations):
            # Extract the submatrix from spdf_combined for this symmetry operation
            # This gives the representation acting on this atom's orbital subspace
            submatrix = spdf_combined[sym_op][idx]
            repr_matrices_for_atom.append(submatrix)

        repr_on_active_orbitals[atom_name] = np.array(repr_matrices_for_atom)

        print(f"Atom {atom_name}: Extracted {num_operations} representation matrices of size {len(active_indices)}x{len(active_indices)}", file=sys.stderr)
    else:
        repr_on_active_orbitals[atom_name] = np.array([])
        print(f"Atom {atom_name}: No active orbitals", file=sys.stderr)


# ==============================================================================
# STEP 11: Verify and report results (debugging output)
# ==============================================================================
for atom_name, repr_matrices in repr_on_active_orbitals.items():
    if repr_matrices.size > 0:
        print(f"\nAtom {atom_name}:", file=sys.stderr)
        print(f"  Number of symmetry operations: {repr_matrices.shape[0]}", file=sys.stderr)
        print(f"  Representation matrix dimension: {repr_matrices.shape[1]}x{repr_matrices.shape[2]}", file=sys.stderr)

        # Get the orbital names for this atom
        active_indices = np.where(atom_orbital_vectors[atom_name] == 1)[0]
        active_orbital_names = [name for name, idx in orbital_map.items() if idx in active_indices]
        print(f"  Active orbitals: {active_orbital_names}", file=sys.stderr)


# ==============================================================================
# STEP 12: Package results and output as JSON
# ==============================================================================
output_data = {
    # Updated orbital vectors (after symmetry completion)
    "updated_orbital_vectors": {name: vec.tolist() for name, vec in atom_orbital_vectors.items()},

    # List of orbitals that were added by symmetry for each atom
    "added_orbitals": added_orbitals_dict,

    # Symmetry representation matrices acting on each atom's active orbital subspace
    "representations_on_active_orbitals": {name: matrices.tolist() for name, matrices in repr_on_active_orbitals.items()}
}

# Output as JSON to stdout
print(json.dumps(output_data), file=sys.stdout)