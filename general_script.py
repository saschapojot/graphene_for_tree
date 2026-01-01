import re
import subprocess
import sys
import os
import json
import numpy as np
from datetime import datetime
from copy import deepcopy
from scipy.linalg import block_diag
import sympy as sp


#this script computes for general

# ==============================================================================
# STEP 1: Validate command line arguments
# ==============================================================================

argErrCode = 20
if (len(sys.argv) != 2):
    print("wrong number of arguments")
    print("example: python preprocessing.py /path/to/mc.conf")
    exit(argErrCode)


confFileName = str(sys.argv[1])

# ==============================================================================
# STEP 2: Parse configuration file
# ==============================================================================
# Run parse_conf.py to read and parse the configuration file
confResult = subprocess.run(
    ["python3", "./parse_files/parse_conf.py", confFileName],
    capture_output=True,
    text=True
)

# Check if the subprocess ran successfully
if confResult.returncode != 0:
    print("Error running parse_conf.py:")
    print(confResult.stderr)
    exit(confResult.returncode)


# Parse the JSON output from parse_conf.py
try:
    parsed_config = json.loads(confResult.stdout)
    # Display parsed configuration in a formatted way
    print("=" * 60)
    print("COMPLETE PARSED CONFIGURATION")
    print("=" * 60)
    # Print basic configuration parameters
    print(f"Name: {parsed_config['name']}")
    print(f"Dimensions: {parsed_config['dim']}")
    print(f"Spin: {parsed_config['spin']}")
    print(f"Neighbors: {parsed_config['neighbors']}")
    print(f"Wyckoff position Type Number: {parsed_config.get('Wyckoff_type_num', 'N/A')}")
    print(f"Lattice Type: {parsed_config.get('lattice_type', 'N/A')}")
    print(f"Space Group: {parsed_config.get('space_group', 'N/A')}")
    # Print space group origin (fractional coordinates)
    origin = parsed_config.get('space_group_origin', [])
    if origin:
        print(f"Space Group Origin: [{', '.join(map(str, origin))}]")

    # Print lattice basis vectors (primitive cell)
    print("Lattice Basis:")
    basis = parsed_config.get('lattice_basis', [])
    for i, vector in enumerate(basis):
        print(f"  Vector {i + 1}: [{', '.join(map(str, vector))}]")

    # Print space group basis vectors
    print("Space Group Basis:")
    sg_basis = parsed_config.get('space_group_basis', [])
    for i, vector in enumerate(sg_basis):
        print(f"  Vector {i + 1}: [{', '.join(map(str, vector))}]")
    # Print Wyckoff Position Types with Orbitals\
    print("\nWyckoff Position Types with Orbitals:")
    wyckoff_types = parsed_config.get('Wyckoff_position_types', {})
    for atom_type, info in wyckoff_types.items():
        print(f"  Type {atom_type}:")
        # Join the list of orbitals into a string
        orbitals_str = ', '.join(info.get('orbitals', []))
        print(f"    Orbitals: {orbitals_str}")
    # Print Wyckoff Positions (Coordinates)
    print("\nWyckoff Positions (Coordinates):")
    wyckoff_positions = parsed_config.get('Wyckoff_positions', [])
    for pos in wyckoff_positions:
        p_name = pos.get('position_name', 'Unknown')
        a_type = pos.get('atom_type', 'Unknown')
        coords = pos.get('fractional_coordinates', [])
        print(f"  {p_name} (atom name: {a_type}):")
        print(f"    Fractional Coords: [{', '.join(map(str, coords))}]")


except json.JSONDecodeError as e:
    print("Failed to parse JSON output from parse_conf.py")
    print(f"Error: {e}")
    print("Raw output:")
    print(confResult.stdout)
    exit(1)

except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit(1)
# Convert parsed_config to JSON string for passing to other subprocesses
config_json = json.dumps(parsed_config)

# ==============================================================================
# STEP 3: Run sanity checks on parsed configuration
# ==============================================================================
print("\n" + "=" * 60)
print("RUNNING SANITY CHECK")
print("=" * 60)


# Run sanity_check.py and pass the JSON data via stdin
sanity_result = subprocess.run(
    ["python3", "./parse_files/sanity_check.py"],
    input=config_json,
    capture_output=True,
    text=True
)
print(f"Exit code: {sanity_result.returncode}")

# Check sanity check results
if sanity_result.returncode != 0:
    print("Sanity check failed!")
    print(f"return code={sanity_result.returncode}")
    print("Error output:")
    print(sanity_result.stderr)
    exit(sanity_result.returncode)
else:
    print("Sanity check passed!")
    print("Output:")
    print(sanity_result.stdout)


# ==============================================================================
# STEP 4: Generate space group representations
# ==============================================================================
print("\n" + "=" * 60)
print("COMPUTING SPACE GROUP REPRESENTATIONS")
print("=" * 60)


# Run generate_space_group_representations.py
sgr_result = subprocess.run(
    ["python3", "./symmetry/generate_space_group_representations.py"],
    input=config_json,
    capture_output=True,
    text=True
)

print(f"Exit code: {sgr_result.returncode}")

# Check if space group representations were generated successfully
if sgr_result.returncode != 0:
    print("Space group representations generation failed!")
    print(f"return code={sgr_result.returncode}")
    print("Error output:")
    print(sgr_result.stderr)
    print("Standard output:")
    print(sgr_result.stdout)
    exit(sgr_result.returncode)


else:
    print("Space group representations generated successfully!")
    # Parse the JSON output
    try:
        space_group_representations = json.loads(sgr_result.stdout)
        print("\n" + "=" * 60)
        print("SPACE GROUP REPRESENTATIONS SUMMARY")
        print("=" * 60)

        # Get number of space group operations
        num_operations = len(space_group_representations["space_group_matrices"])
        print(f"Number of space group operations: {num_operations}")
        # Print space group origin in different coordinate systems

        print("\nSpace Group Origin:")
        origin_cart = space_group_representations["space_group_origin_cartesian"]
        origin_cart=np.array(origin_cart)
        origin_frac_prim = space_group_representations["space_group_origin_fractional_primitive"]
        print(
            f"  Bilbao (fractional in space group basis): [{', '.join(map(str, parsed_config['space_group_origin']))}]")
        print(f"  Cartesian: [{', '.join(f'{x:.6f}' for x in origin_cart)}]")
        print(f"  Fractional (primitive cell basis): [{', '.join(f'{x:.6f}' for x in origin_frac_prim)}]")

        # Extract orbital representations (s, p, d, f)
        repr_s, repr_p, repr_d, repr_f = space_group_representations["repr_s_p_d_f"]

        # Print dimensions of representation matrices
        print(f"\nOrbital Representations:")
        print(f"  s orbitals: {len(repr_s)} operations × {len(repr_s[0])}×{len(repr_s[0][0])} matrices")
        print(f"  p orbitals: {len(repr_p)} operations × {len(repr_p[0])}×{len(repr_p[0][0])} matrices")
        print(f"  d orbitals: {len(repr_d)} operations × {len(repr_d[0])}×{len(repr_d[0][0])} matrices")
        print(f"  f orbitals: {len(repr_f)} operations × {len(repr_f[0])}×{len(repr_f[0][0])} matrices")

        # Convert to NumPy arrays for further processing
        space_group_matrices = np.array(space_group_representations["space_group_matrices"])
        space_group_matrices_cartesian = np.array(space_group_representations["space_group_matrices_cartesian"])
        space_group_matrices_primitive = np.array(space_group_representations["space_group_matrices_primitive"])
        #  space group matrices in Cartesian coordinates , a list
        space_group_bilbao_cart = [np.array(item) for item in space_group_matrices_cartesian]
        repr_s_np = np.array(repr_s)
        repr_p_np = np.array(repr_p)
        repr_d_np = np.array(repr_d)
        repr_f_np = np.array(repr_f)
        print("\nSpace group representations loaded and converted to NumPy arrays.")
        print(f"Available matrices:")
        print(f"  - space_group_matrices: {space_group_matrices.shape}")
        print(f"  - space_group_matrices_cartesian: {space_group_matrices_cartesian.shape}")
        print(f"  - space_group_matrices_primitive: {space_group_matrices_primitive.shape}")
        print(f"  - s orbital representations: {repr_s_np.shape}")
        print(f"  - p orbital representations: {repr_p_np.shape}")
        print(f"  - d orbital representations: {repr_d_np.shape}")
        print(f"  - f orbital representations: {repr_f_np.shape}")
    except json.JSONDecodeError as e:
        print("Error parsing JSON output from space group representations:")
        print(f"JSON Error: {e}")
        print("Raw output was:")
        print(sgr_result.stdout)
        exit(1)
    except KeyError as e:
        print(f"Missing key in space group representations output: {e}")
        print("Available keys:", list(
            space_group_representations.keys()) if 'space_group_representations' in locals() else "Could not parse JSON")
        exit(1)
lattice_basis = np.array(parsed_config['lattice_basis'])

# ==============================================================================
# STEP 5: Define orbital mapping for 78-dimensional orbital space
# ==============================================================================
# Maps orbital names (like '3dxy') to their index in the orbital vector
# Total: 78 orbitals from 1s to 7f
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
# STEP 6: Complete orbital basis under symmetry operations
# ==============================================================================


print("\n" + "=" * 60)
print("COMPLETING ORBITALS UNDER SYMMETRY")
print("=" * 60)

# Combine parsed_config and space_group_representations
combined_input = {
    "parsed_config": parsed_config,
    "space_group_representations": space_group_representations
}


# Convert to JSON for subprocess
combined_input_json = json.dumps(combined_input)

# Run complete_orbitals.py
completing_result = subprocess.run(
    ["python3", "./symmetry/complete_orbitals.py"],
    input=combined_input_json,
    capture_output=True,
    text=True
)

# Check if orbital completion succeeded
if completing_result.returncode != 0:
    print("Orbital completion failed!")
    print(f"Return code: {completing_result.returncode}")
    print("Error output:")
    print(completing_result.stderr)
    exit(completing_result.returncode)


# Parse the output
try:
    orbital_completion_data = json.loads(completing_result.stdout)
    print("Orbital completion successful!")
    # Display which orbitals were added by symmetry
    print("\n" + "-" * 40)
    print("ORBITALS ADDED BY SYMMETRY:")
    print("-" * 40)
    added_orbitals = orbital_completion_data["added_orbitals"]

    if any(added_orbitals.values()):
        for atom_type, orbitals in added_orbitals.items():
            if orbitals:
                print(f"  {atom_type}: {', '.join(orbitals)}")
    else:
        print("  No additional orbitals needed - input was already complete")

    # Display final active orbitals for each atom
    print("\n" + "-" * 40)
    print("FINAL ACTIVE ORBITALS PER ATOM:")
    print("-" * 40)

    updated_vectors = orbital_completion_data["updated_orbital_vectors"]

    orbital_map_reverse = {v: k for k, v in orbital_map.items()}  # Reverse lookup
    for atom_type, vector in updated_vectors.items():
        # Find indices where orbital is active (value = 1)
        active_indices = [i for i, val in enumerate(vector) if val == 1]
        # Convert indices back to orbital names
        active_orbital_names = [orbital_map_reverse.get(idx, f"unknown_{idx}") for idx in active_indices]
        print(f"  {atom_type} ({len(active_orbital_names)} orbitals): {', '.join(active_orbital_names)}")
    # Display symmetry representation information
    print("\n" + "-" * 40)
    print("SYMMETRY REPRESENTATIONS ON ACTIVE ORBITALS:")
    print("-" * 40)
    representations = orbital_completion_data["representations_on_active_orbitals"]
    for atom_type, repr_matrices in representations.items():
        if repr_matrices:
            repr_array = np.array(repr_matrices)
            print(
                f"  {atom_type}: {repr_array.shape[0]} operations, {repr_array.shape[1]}×{repr_array.shape[2]} matrices")
    # Update parsed_config with completed orbitals
    for atom_pos in parsed_config['Wyckoff_positions']:
        position_name = atom_pos['position_name']
        # print(f"position_name={position_name}")
        atom_type = atom_pos['atom_type']
        # Get the updated orbital vector for this atom
        # print(f"updated_vectors={updated_vectors}")
        # print(f"position_name={position_name}")

        if position_name in updated_vectors:
            vector = updated_vectors[position_name]
            active_indices = [i for i, val in enumerate(vector) if val == 1]
            # print(f"active_indices={active_indices}")
            active_orbital_names = [orbital_map_reverse.get(idx) for idx in active_indices]
            # print(f"active_orbital_names={active_orbital_names}")
            parsed_config['Wyckoff_position_types'][position_name]['orbitals'] = active_orbital_names
            parsed_config['Wyckoff_position_types'][position_name]['orbitals_completed'] = True

    # Store completion results for later use
    orbital_completion_results = {
        "status": "completed",
        "added_orbitals": added_orbitals,
        "orbital_vectors": updated_vectors,
        "representations_on_active_orbitals": representations,
    }

except json.JSONDecodeError as e:
    print("Error parsing JSON output from complete_orbitals.py:")
    print(f"JSON Error: {e}")
    print("Raw output:")
    print(completing_result.stdout)
    print("Error output:")
    print(completing_result.stderr)
    exit(1)

except KeyError as e:
    print(f"Missing key in orbital completion output: {e}")
    print("Available keys:",
          list(orbital_completion_data.keys()) if 'orbital_completion_data' in locals() else "Could not parse JSON")
    exit(1)


except Exception as e:
    print(f"Unexpected error processing orbital completion: {e}")
    print("Type:", type(e).__name__)
    exit(1)

print("\n" + "=" * 60)
print("ORBITAL COMPLETION FINISHED")
print("=" * 60)

# ==============================================================================
# Helper function: orbital_to_submatrix
# ==============================================================================
def orbital_to_submatrix(orbitals, Vs, Vp, Vd, Vf):
    """
    Extract submatrix from full orbital representation for specific orbitals

    Args:
        orbitals: List of orbital names (e.g., ['2s', '2px', '2py', '2pz'])
        Vs, Vp, Vd, Vf: Representation matrices for s, p, d, f orbitals

    Returns:
        numpy array: Submatrix for the specified orbitals
    """
    full_orbitals = [
        's',
        'px', 'py', 'pz',
        'dxy', 'dyz', 'dzx', 'd(x2-y2)', 'd(3z2-r2)',
        'fz3', 'fxz3', 'fyz3', 'fxyz', 'fz(x2-y2)', 'fx(x2-3y2)', 'fy(3x2-y2)'
    ]
    # Remove leading numbers from orbitals (e.g., '2s' -> 's', '2pz' -> 'pz')
    orbital_types = []
    for orb in orbitals:
        # Remove all leading digits
        orbital_type = orb.lstrip('0123456789')
        orbital_types.append(orbital_type)
    # Sort orbitals by their position in full_orbitals
    sorted_orbital_types = sorted(orbital_types, key=lambda orb: full_orbitals.index(orb))
    # Get the indices in full_orbitals
    orbital_indices = [full_orbitals.index(orb) for orb in sorted_orbital_types]
    # Build full representation matrix
    hopping_matrix_full = block_diag(Vs, Vp, Vd, Vf)
    # Extract submatrix for the specific orbitals
    V_submatrix = hopping_matrix_full[np.ix_(orbital_indices, orbital_indices)]
    return V_submatrix


# ==============================================================================
# atomIndex class with orbital representations
# ==============================================================================
def frac_to_cartesian(cell, frac_coord, basis,origin_cart):
    """
    Convert fractional coordinates to Cartesian coordinates.
    The transformation is:
        r_cart = (n0 + f0) * a0 + (n1 + f1) * a1 + (n2 + f2) * a2 + origin
    where:
        - [n0, n1, n2] are unit cell indices
        - [f0, f1, f2] are fractional coordinates within the cell
        - [a0, a1, a2] are lattice basis vectors
        - origin is the coordinate system origin (e.g., Bilbao origin)
    Args:
        cell: [n0, n1, n2] unit cell indices
        frac_coord:  [f0, f1, f2] fractional coordinates within the unit cell
        basis: lattice basis vectors, 3×3 array where each row is a basis vector
        origin_cart: Origin offset in Cartesian coordinates, For Bilbao convention, use space_group_origin_cartesian

    Returns:
        numpy array: Cartesian coordinates (3D vector)
    """
    n0, n1, n2 = cell
    f0, f1, f2 = frac_coord
    a0, a1, a2 = basis
    # Compute Cartesian position from fractional coordinates
    r_cart = (n0 + f0) * a0 + (n1 + f1) * a1 + (n2 + f2) * a2+ np.array(origin_cart)
    return r_cart


class atomIndex:
    def __init__(self, cell, frac_coord,position_name, basis,origin_cart, parsed_config,
                 repr_s_np, repr_p_np, repr_d_np, repr_f_np):
        """
        Initialize an atom with position, orbital, and representation information
        Args:
            cell: [n0, n1, n2] unit cell indices
            frac_coord: [f0, f1, f2] fractional coordinates
            position_name: specific atom identifier for Wyckoff positions (e.g., 'C0', 'C1')
            basis: lattice basis vectors [a0, a1, a2], stored as row vectors
            origin_cart: Cartesian origin offset (e.g., Bilbao origin)
            parsed_config: configuration dict containing orbital information
            repr_s_np, repr_p_np, repr_d_np, repr_f_np: representation matrices for s,p,d,f orbitals
        """
        # Deep copy mutable inputs
        self.n0 = deepcopy(cell[0])
        self.n1 = deepcopy(cell[1])
        self.n2 = deepcopy(cell[2])
        self.position_name = position_name  # Specific identifier for Wyckoff position (e.g., 'C0')

        #example: C00,C01,
        #format: [Species][Wyckoff][Index]
        # the first 0 refers to a unique Wyckoff position,
        # there may be multiple atoms for this Wyckoff position,
        # the second 0,1 distinguish these atoms on the same Wyckoff position
        #the value will be set later
        self.wyckoff_instance_id=None
        self.frac_coord = deepcopy(frac_coord)
        self.basis = deepcopy(basis)
        self.origin_cart = deepcopy(origin_cart)  # ← Store origin!
        self.parsed_config = deepcopy(parsed_config)

        # Calculate Cartesian coordinates using frac_to_cartesian helper
        # The basis vectors a0, a1, a2 are primitive lattice vectors expressed in
        # Cartesian coordinates using Bilbao's origin, so the result is
        # Cartesian coordinates using Bilbao's origin
        self.cart_coord = frac_to_cartesian(cell, frac_coord, basis,origin_cart)

        # parse Configuration to find orbitals
        found_atom = False
        self.orbitals = []
        self.atom_type = None  # To be determined from config

        wyckoff_positions = parsed_config['Wyckoff_positions']
        wyckoff_types = parsed_config['Wyckoff_position_types']

        # Lookup by position_name (e.g., 'C0')
        for pos in wyckoff_positions:
            if pos.get('position_name') == position_name:
                # Retrieve the atom type defined in the config (e.g., 'C')
                self.atom_type = pos.get('atom_type')
                # Directly access the orbital definition
                self.orbitals = deepcopy(wyckoff_types[self.position_name]['orbitals'])
                found_atom = True
                break
        if not found_atom:
            raise ValueError(f"Configuration not found for position '{position_name}'")

        self.num_orbitals = len(self.orbitals)
        # Deep copy representation matrices (all required now)
        self.repr_s_np = deepcopy(repr_s_np)
        self.repr_p_np = deepcopy(repr_p_np)
        self.repr_d_np = deepcopy(repr_d_np)
        self.repr_f_np = deepcopy(repr_f_np)

        # Pre-compute representation matrices for this atom's orbitals
        self.orbital_representations = None
        self._compute_orbital_representations()

    def _compute_orbital_representations(self):
        """
        Pre-compute orbital representation matrices for all space group operations
        Returns a list where each element is the representation matrix for one operation
        """
        num_operations = len(self.repr_s_np)
        self.orbital_representations = []
        for op_idx in range(num_operations):
            Vs = self.repr_s_np[op_idx]
            Vp = self.repr_p_np[op_idx]
            Vd = self.repr_d_np[op_idx]
            Vf = self.repr_f_np[op_idx]
            # Get submatrix for this atom's specific orbitals
            V_submatrix = orbital_to_submatrix(self.orbitals, Vs, Vp, Vd, Vf)
            self.orbital_representations.append(V_submatrix)

    def get_representation_matrix(self, operation_idx):
        """
        Get the orbital representation matrix for a specific space group operation

        Args:
            operation_idx: index of the space group operation

        Returns:
            numpy array: representation matrix for this atom's orbitals
        """
        if self.orbital_representations is None:
            raise ValueError(f"Orbital representations not computed for atom {self.atom_type}")

        if operation_idx >= len(self.orbital_representations):
            raise IndexError(f"Operation index {operation_idx} out of range")

        return self.orbital_representations[operation_idx]

    def get_sympy_representation_matrix(self, operation_idx):
        """
        Get the orbital representation matrix as a sympy Matrix

        Args:
            operation_idx: index of the space group operation

        Returns:
            sympy.Matrix: representation matrix for this atom's orbitals
        """
        return sp.Matrix(self.get_representation_matrix(operation_idx))

    def __str__(self):
        """String representation for print()"""
        orbitals_str = ', '.join(self.orbitals) if self.orbitals else "None"
        repr_info = f", Repr: ✓" if self.orbital_representations is not None else ""
        return (f"Atom: {self.atom_type}, "
                f"position_name='{self.position_name}', "
                f"wyckoff_instance_id='{self.wyckoff_instance_id}', "
                f"Cell: [{self.n0}, {self.n1}, {self.n2}], "
                f"Frac: {self.frac_coord}, "
                f"Cart: {self.cart_coord}, "
                f"Orbitals: [{orbitals_str}]"
                f"{repr_info}"
                )

    def __repr__(self):
        """Detailed representation for debugging"""
        orbitals_str = ', '.join(self.orbitals) if self.orbitals else "None"
        return (f"atomIndex(cell=[{self.n0}, {self.n1}, {self.n2}], "
                f"frac_coord={self.frac_coord}, "
                f"atom_type='{self.atom_type}', "
                f"position_name='{self.position_name}', "
                 f"wyckoff_instance_id='{self.wyckoff_instance_id}', "
                f"orbitals=[{orbitals_str}])")

    def get_orbital_names(self):
        """Get list of orbital names for this atom"""
        return self.orbitals

    def has_orbital(self, orbital_name):
        """Check if this atom has a specific orbital"""
        # Handle both '2s' and 's' format
        orbital_type = orbital_name.lstrip('0123456789')
        return any(orb.lstrip('0123456789') == orbital_type for orb in self.orbitals)



def compute_dist(center_atom, unit_cell_atoms, search_range=10, radius=None, search_dim=2):
    """
    Find all atoms within a specified radius of a center atom by searching neighboring cells.
    Returns constructed atomIndex objects for all neighbors found. The neighboring atom types are determined by
    unit_cell_atoms
    Args:
        center_atom: atomIndex object for the center atom
        unit_cell_atoms: list of atomIndex objects in the reference unit cell [0,0,0]
        search_range: how many cells to search in each direction (default: 1)
        radius: cutoff distance in Cartesian coordinates (default: None means all atoms)
        search_dim: dimension to search (1, 2, or 3) (default: 2)
            - 1: search along n0 only
            - 2: search along n0 and n1
            - 3: search along n0, n1, and n2

    Returns:
        list: atomIndex objects within the specified radius, sorted by distance
    """
    neighbor_atoms = []
    center_cart = center_atom.cart_coord
    lattice_basis = center_atom.basis
    origin_cart = center_atom.origin_cart  # ← Get origin from center atom

    # Determine search ranges based on search_dim
    if search_dim == 1:
        n0_range = range(-search_range, search_range + 1)
        n1_range = [0]
        n2_range = [0]
    elif search_dim == 2:
        n0_range = range(-search_range, search_range + 1)
        n1_range = range(-search_range, search_range + 1)
        n2_range = [0]
    else:  # search_dim == 3
        n0_range = range(-search_range, search_range + 1)
        n1_range = range(-search_range, search_range + 1)
        n2_range = range(-search_range, search_range + 1)

    # Search through neighboring cells
    for n0 in n0_range:
        for n1 in n1_range:
            for n2 in n2_range:
                cell = [n0, n1, n2]

                # Check each atom in the unit cell
                for unit_atom in unit_cell_atoms:
                    # Compute Cartesian coordinates for this atom in the proposed cell
                    candidate_cart = frac_to_cartesian(cell, unit_atom.frac_coord, lattice_basis,origin_cart)

                    # Calculate distance
                    dist = np.linalg.norm(candidate_cart - center_cart)

                    # Only construct atom if it passes the distance check
                    if radius is None or dist <= radius:
                        # Create atomIndex for this atom in the current cell with deep copies
                        neighbor_atom = atomIndex(
                            cell=deepcopy(cell),
                            frac_coord=deepcopy(unit_atom.frac_coord),
                            position_name=unit_atom.position_name,
                            basis=deepcopy(lattice_basis),
                            origin_cart=deepcopy(origin_cart),
                            parsed_config=deepcopy(unit_atom.parsed_config),
                            repr_s_np=deepcopy(unit_atom.repr_s_np),
                            repr_p_np=deepcopy(unit_atom.repr_p_np),
                            repr_d_np=deepcopy(unit_atom.repr_d_np),
                            repr_f_np=deepcopy(unit_atom.repr_f_np)
                        )
                        neighbor_atom.wyckoff_instance_id=unit_atom.wyckoff_instance_id

                        # Deep copy orbital information from unit cell atom
                        # neighbor_atom.orbitals = deepcopy(unit_atom.orbitals)
                        # neighbor_atom.num_orbitals = unit_atom.num_orbitals
                        # neighbor_atom.orbital_representations = deepcopy(unit_atom.orbital_representations)

                        neighbor_atoms.append((dist, neighbor_atom))

    # Sort by distance and return only the atomIndex objects
    neighbor_atoms.sort(key=lambda x: x[0])
    return [atom for dist, atom in neighbor_atoms]


def get_rotation_translation(space_group_bilbao_cart, operation_idx):
    """
    Extract rotation/reflection matrix R and translation vector t from a space group operation.

    The space group operation is in the form [R|t], represented as a 3×4 matrix:
        [R | t] = [R00 R01 R02 | t0]
                  [R10 R11 R12 | t1]
                  [R20 R21 R22 | t2]

    The operation transforms a position vector r as: r' = R @ r + t

    Args:
        space_group_bilbao_cart: List of space group matrices in Cartesian coordinates
                                 using Bilbao origin (shape: num_ops × 3 × 4)
        operation_idx: Index of the space group operation

    Returns:
        tuple: (R, t)
            - R (ndarray): 3×3 rotation/reflection matrix
            - t (ndarray): 3D translation vector
    """
    operation = space_group_bilbao_cart[operation_idx]
    R = operation[:3, :3]  # Rotation/reflection part
    t = operation[:3, 3]  # Translation part

    return R, t


def find_identity_operation(space_group_bilbao_cart, tolerance=1e-9, verbose=True):
    """
    Find the index of the identity operation in space group matrices.

    The identity operation has:
    - Rotation part: 3×3 identity matrix
    - Translation part: zero vector

    Args:
        space_group_bilbao_cart: List or array of  3×4 space group matrices [R|t]
                                 in Cartesian coordinates
        tolerance: Numerical tolerance for comparison (default: 1e-9)
        verbose: Whether to print status messages (default: True)

    Returns:
        int: Index of the identity operation

    Raises:
        ValueError: If identity operation is not found
    """
    identity_idx = None

    for idx in range(len(space_group_bilbao_cart)):
        # Extract rotation and translation using helper function
        R, t = get_rotation_translation(space_group_bilbao_cart, idx)

        # Check if rotation is identity and translation is zero
        if np.allclose(R, np.eye(3), atol=tolerance) and \
                np.allclose(t, np.zeros(3), atol=tolerance):
            identity_idx = idx
            break

    if identity_idx is None:
        error_msg = "Identity operation not found in space_group_bilbao_cart!"
        raise ValueError(error_msg)

    return identity_idx


# ==============================================================================
# hopping class
# ==============================================================================
class hopping:
    """
    Represents a single hopping term from a neighbor atom to a center atom.
    The hopping direction is: to_atom (center) ← from_atom (neighbor)
    The hopping is defined by a space group operation that transforms a seed hopping.
    This hopping is obtained from seed hopping by transformation:
    r' = R @ r + t + n₀·a₀ + n₁·a₁ + n₂·a₂
    where R is rotation, t is translation, and n_vec = [n₀, n₁, n₂] is the lattice shift,
    and r is the position vector from seed hopping's from_atom (neighbor) to to_atom (center).
    """


    def __init__(self, to_atom, from_atom, operation_idx, rotation_matrix, translation_vector, n_vec, is_seed):
        """
        Initialize a hopping term: to_atom (center) ← from_atom (neighbor).
         This hopping is generated by applying a space group operation to a seed hopping.
         The transformation maps the seed neighbor position to this hopping's neighbor position.

        :param to_atom: atomIndex object for the center atom (hopping destination)
        :param from_atom: atomIndex object for the neighbor atom (hopping source)

        :param operation_idx: Index of the space group operation that generates this hopping
                          from the seed hopping in the equivalence class
        :param rotation_matrix: 3×3 rotation/reflection matrix R (in Cartesian coordinates, Bilbao origin)
        :param translation_vector: 3D translation vector t from the Bilbao space group operation
                              (in Cartesian coordinates, Bilbao origin)
        :param n_vec: Array [n₀, n₁, n₂] containing integer coefficients for lattice translation
                  This is the additional lattice shift that is not given by Bilbao data
                  The full transformation is: r' = R @ r + t + n₀·a₀ + n₁·a₁ + n₂·a₂
                  Note that Bilbao only gives R and t
        :param is_seed:  Boolean flag indicating if this is the seed hopping for its equivalence class
                    True for the seed hopping (generated by identity operation)
                    False for derived hoppings (generated by other symmetry operations)
        """

        self.to_atom = deepcopy(to_atom)  # Deep copy of center atom (destination)
        self.from_atom = deepcopy(from_atom)   # Deep copy of neighbor atom (source)
        self.operation_idx = operation_idx  # Which space group operation transforms parent hopping to this hopping
        self.rotation_matrix = deepcopy(rotation_matrix)  # Deep copy of 3×3 Bilbao rotation  matrix R
        self.translation_vector = deepcopy(translation_vector)# Deep copy of 3D Bilbao translation t
        self.n_vec=np.array(n_vec)  # Lattice translation coefficients [n₀, n₁, n₂]
                                      # Additional lattice shift not given by Bilbao data
                                      # Computed to preserve center atom invariance
        self.is_seed=is_seed # Boolean: True if this is the seed hopping, False if derived from seed (parent)
        self.distance = None  # Euclidean distance between center (to_atom) and neighbor (from_atom)
        self.T = None  # Hopping matrix between orbital basis (sympy Matrix, to be computed)
                       # Represents the tight-binding hopping matrix: center orbitals ← neighbor orbitals

    def conjugate(self):
        """
        Return the conjugate (reverse) hopping direction.
         For this hopping: center ← neighbor, the conjugate is: neighbor ← center.
         This is used to enforce Hermiticity constraints in tight-binding models:
         T(neighbor ← center) = T(center ← neighbor)†
        :return: list: [from_atom, to_atom] with swapped order (deep copied)
                 Represents the reverse hopping: neighbor ← center
        """
        return [deepcopy(self.from_atom), deepcopy(self.to_atom)]

    def compute_distance(self):
        """
         Compute the Euclidean distance from the neighbor atom to the center atom.
         This distance is calculated in Cartesian coordinates using Bilbao origin.
         All hoppings in the same equivalence class should have the same distance
         (up to numerical precision), as they are related by symmetry operations.
        Adds member variable self.distance: L2 norm of the position difference vector (center - neighbor)
        """
        pos_to = self.to_atom.cart_coord  # Cartesian position of center atom
        pos_from = self.from_atom.cart_coord # Cartesian position of neighbor atom

        # Real space position difference vector (center - neighbor)
        delta_pos = pos_to - pos_from
        # Compute Euclidean distance (L2 norm)
        self.distance = np.linalg.norm(delta_pos, ord=2)

    def __repr__(self):
        """
        String representation for debugging and display.

        :return: str: Compact representation showing: center_type[n0,n1,n2] ← neighbor_type[m0,m1,m2],
                 operation index, distance, and seed status
        """
        seed_marker = " [SEED]" if self.is_seed else ""
        distance_str = f"{self.distance:.4f}" if self.distance is not None else "None"

        # Format cell indices for to_atom and from_atom
        to_cell = f"[{self.to_atom.n0},{self.to_atom.n1},{self.to_atom.n2}]"
        from_cell = f"[{self.from_atom.n0},{self.from_atom.n1},{self.from_atom.n2}]"

        return (f"hopping({self.to_atom.atom_type}{to_cell} ← {self.from_atom.atom_type}{from_cell}, "
                f"op={self.operation_idx}, "
                f"distance={distance_str}"
                f"{seed_marker})")


# ==============================================================================
# vertex class
# ==============================================================================

class vertex():
    """
    Represents a node in the symmetry constraint tree for tight-binding hopping matrices.
    Each vertex contains a hopping object, the hopping object contains hopping matrix of to_atom (center) ← from_atom (neighbor)
    The tree structure represents how parent hopping generates this hopping by space group operations or Hermiticity constraints.

    Tree Structure:
      - Root vertex: Corresponds to the seed hopping (identity operation)
      - Child vertices: Hoppings derived from parent through symmetry operations or Hermiticity
      - Constraint types: "linear" (from space group) or "hermitian" (from H† = H)
    The tree is used to:
     1. Express derived hopping matrices in terms of independent matrices (in root)
     2. Enforce symmetry constraints automatically
     3. Reduce the number of independent tight-binding parameters

     CRITICAL: Tree Structure Uses References (Pointers)
     ================================================
     The parent-child relationships are implemented using REFERENCES (C++ sense) / POINTERS (C sense):
     - self.parent stores a REFERENCE to the parent vertex object (not a copy)
     - self.children stores a list of REFERENCES to child vertex objects (not copies)

     This means:
     - Multiple vertices can reference the same parent object
     - Modifying a parent's hopping matrix T affects all children's constraint calculations
     - The tree forms a true graph structure in memory with shared nodes
     - Deleting a vertex requires careful handling to avoid dangling references


     Memory Diagram Example:
     ----------------------
     Root Vertex (id=0x1000) ──┬──> Child 1, linear (address=0x2000, parent address=0x1000)
                               ├──> Child 2, linear (address=0x3000, parent address=0x1000)
                               └──> Child 3, hermitian (address=0x4000, parent address=0x1000)
    All three children have parent=0x1000 (same memory address)
    Root's self.children = [0x2000, 0x3000, 0x4000] (references, not copies)
    """

    def __init__(self, hopping, type, identity_idx, parent=None):
        """
        Initialize a vertex in the tree.
        Args:
            hopping: hopping object representing the tight-binding term: center ← neighbor
            Contains the hopping matrix T between orbital basis,
            T's row represents: center atom orbitals
            T's column represents: neighbor atom orbitals
            one element in T is the hopping coefficient from one orbital in neighbor atom to
             one orbital in center atom
            type: Constraint type that shows how this vertex is derived from its parent
                   - "linear": Derived from parent via space group symmetry operation
                   - "hermitian": Derived from parent via Hermiticity constraint
                   - None: It is root vertex
            identity_idx: Index of the identity operation in space_group_bilbao_cart
                        Used to identify root vertices (hopping.operation_idx == identity_idx)
            parent: REFERENCE to parent vertex object (default: None for root)
                    NOT deep copied - this is a reference (C++ sense) / pointer (C sense)

                    Why parent is a reference:
                     -------------------------
                     1. Upward Traversal: Allows child → parent → root navigation
                     2. Constraint Access: Child can read parent's hopping matrix T
                     3. Shared Parent: Multiple children reference same parent object
                     IMPORTANT: parent=None only for root vertices
                                parent≠None for all derived vertices (children)


        """
        self.hopping = deepcopy(hopping) # Deep copy of hopping object containing:
                                         # - to_atom (center), from_atom (neighbor)
                                         # - is_seed, operation_idx
                                        # - rotation_matrix R, translation_vector t, n_vec
                                        # - distance, T (hopping matrix)

        self.type = type # Constraint type: None (root), "linear" (symmetry), or "hermitian"
                         # String is immutable, safe to assign directly
        self.is_root = (hopping.operation_idx == identity_idx)  # Boolean flag identifying root vertex
                                                                # Root vertex contains identity operation
                                                                # Starting vertex of hopping matrix T propagation

        self.children = []  # List of REFERENCES to child vertex objects
                            # CRITICAL: These are references (pointers), NOT deep copies!
                            #
                            # Why references are essential:
                            # -----------------------------
                            # 1. Tree Structure: Forms true parent-child graph in memory
                            # 2. Constraint Propagation: Changes to root's T affect tree traversal
                            # 3. Memory Efficiency: Avoids duplicating entire subtrees
                            # 4. Bidirectional Links: Children can access parent via self.parent
                            #
                            # Usage:
                            # ------
                            # - Empty list [] at initialization (no children yet)
                            # - Populated via add_child() method with vertex references
                            # - Each element points to a vertex object in memory
                            #
                            # WARNING: Do NOT deep copy children when copying a vertex!
                            #          This would break the tree structure.


        self.parent = parent  # Reference to parent vertex (None for root)
                              # NOT deep copied, because this is reference (reference in C++ sense, pointer in C sense)
                              # Forms bidirectional directed tree: parent ↔ children

    def add_child(self, child_vertex):
        """
        Add a child vertex to this vertex and set bidirectional parent-child relationship.

        CRITICAL: Reference-Based Tree Construction
        ===========================================
        This method establishes bidirectional links using REFERENCES (pointers):
        Before call:
        -----------
        self (parent vertex at address 0x1000):
            self.children = [0x2000, 0x3000]  # existing children
        child_vertex (at address 0x4000):
            child_vertex.parent = None  # or some other parent #or this child is a root, we are adding a subtree

        After self.add_child(child_vertex):
        -----------------------------------
        self (parent vertex at address 0x1000):
            self.children = [0x2000, 0x3000, 0x4000]  # added reference 0x4000
        child_vertex (at address 0x4000):
            child_vertex.parent = 0x1000  # reference to self

        Args:
             child_vertex: vertex object to add as a child
                           The child represents a hopping derived from this vertex's hopping
                           either through symmetry operation (type="linear")
                           or Hermiticity (type="hermitian")

                           IMPORTANT: child_vertex is NOT deep copied
                                      The REFERENCE to child_vertex is stored in self.children

        Returns:
                None (modifies self.children and child_vertex.parent in-place)
        """
        self.children.append(child_vertex) # Add REFERENCE to child_vertex to this vertex's children list
                                           # NOT a deep copy - the actual vertex object reference
                                           # After this: self.children[-1] is child_vertex (same object)
                                           #
                                           # Memory effect:
                                           # - self.children list grows by 1 element
                                           # - That element is a reference (memory address) to child_vertex
                                           # - No new vertex object is created




        child_vertex.parent = self  # Set bidirectional relationship: this vertex becomes the child's parent
                                    # Stores new vertex parent's REFERENCE (C++ sense) / POINTER (C sense) to the new vertex
                                    # NOT a deep copy - the actual parent vertex object reference
                                    # After this: child_vertex.parent is self (same object)
                                    #
                                    # Memory effect:
                                    # - child_vertex.parent now points to self's memory address
                                    # - Creates upward link in tree: child → parent
                                    # - Combined with append above: creates bidirectional edge
                                    # WARNING: This overwrites any previous parent!

    def __repr__(self):
        """
        String representation for debugging and display.
        Shows the vertex's role in the tree (ROOT or CHILD), constraint type,
        operation index, parent information, and number of children.
        Returns: str: Compact representation showing vertex type, operation, parent, and children count
                      Format: "vertex(type=<type>, <ROOT/CHILD>, op=<op_idx>, parent=<parent_info>, children=<count>)"

        """
        # Determine if this is a root or child vertex
        root_str = "ROOT" if self.is_root else "CHILD"

        # Show parent's operation index if parent exists, otherwise "None"
        # Parent is None for root vertices
        parent_str = "None" if self.parent is None else f"op={self.parent.hopping.operation_idx}"
        # Return formatted string with key vertex information:
        # - type: constraint type (None, "linear", or "hermitian")
        # - ROOT/CHILD: vertex role in tree
        # - op: this vertex's space group operation index
        # - parent: parent's operation index or "None"
        # - children: number of child vertices
        return (f"vertex(type={self.type}, {root_str}, "
                f"op={self.hopping.operation_idx}, "
                f"parent={parent_str}, "
                f"children={len(self.children)})")

def is_lattice_vector(vector, lattice_basis, tolerance=1e-5):
    """
    Check if a vector can be expressed as an integer linear combination of lattice basis vectors.

    A vector v is a lattice vector if:
        v = n0*a0 + n1*a1 + n2*a2
    where n0, n1, n2 are integers and a0, a1, a2 are primitive lattice basis vectors.

    Args:
        vector: 3D vector to check (Cartesian coordinates)
        lattice_basis: Primitive lattice basis vectors (3×3 array, each row is a basis vector)
                      expressed in Cartesian coordinates using Bilbao origin
        tolerance: Numerical tolerance for checking if coefficients are integers (default: 1e-5)

    Returns:
        tuple: (is_lattice, n_vector)
            - is_lattice (bool): True if vector is a lattice vector
            - n_vector (ndarray): The integer coefficients [n0, n1, n2]
    """
    # Extract basis vectors (each row is a basis vector)
    a0, a1, a2 = lattice_basis

    # Create matrix with basis vectors as columns
    lattice_matrix = np.column_stack([a0, a1, a2])

    # Solve: vector = lattice_matrix @ [n0, n1, n2]
    # So: [n0, n1, n2] = lattice_matrix^(-1) @ vector
    n_vector_float = np.linalg.solve(lattice_matrix, vector)

    # Round to nearest integers
    n_vector = np.round(n_vector_float)

    # Check if coefficients are integers (within tolerance)
    is_lattice = np.allclose(n_vector_float, n_vector, atol=tolerance)

    return is_lattice, n_vector


def check_center_invariant(center_atom, operation_idx, space_group_bilbao_cart,
                           lattice_basis, tolerance=1e-5, verbose=False):
    """
    Check if a center atom is invariant under a specific space group operation.

    An atom is invariant if the symmetry operation maps it to itself, possibly
    translated by a lattice vector. The actual operation is:
        r' = R @ r + t + n0*a0 + n1*a1 + n2*a2
    where n0, n1, n2 are integers and a0, a1, a2 are primitive lattice basis vectors.

    For invariance, we need: r' = r, which means:
        R @ r + t + n0*a0 + n1*a1 + n2*a2 = r
        => (R - I) @ r + t = -(n0*a0 + n1*a1 + n2*a2)

    Args:
        center_atom: atomIndex object representing the center atom
        operation_idx: Index of the space group operation to check
        space_group_bilbao_cart: List of space group matrices in Cartesian coordinates
                                 using Bilbao origin (shape: num_ops × 3 × 4)
        lattice_basis: Primitive lattice basis vectors (3×3 array, each row is a basis vector)
                      expressed in Cartesian coordinates using Bilbao origin
        tolerance: Numerical tolerance for comparison (default: 1e-5)
        verbose: Whether to print debug information (default: False)

    Returns:
        tuple: (is_invariant, n_vector)
            - is_invariant (bool): True if the atom is invariant under the operation
            - n_vector (ndarray): The integer coefficients [n0, n1, n2] for lattice translation
    """
    # Extract the rotation matrix R and translation vector t from the space group operation
    R, t = get_rotation_translation(space_group_bilbao_cart, operation_idx)

    # Get center atom's Cartesian position (using Bilbao origin)
    r_center = center_atom.cart_coord

    # Compute the position after applying only R and t (without lattice translation yet)
    # This is: R @ r + t
    # r_transformed = R @ r_center + t

    # Compute the left-hand side of the invariance equation:
    # (R - I) @ r + t
    # For invariance, this must equal -(n0*a0 + n1*a1 + n2*a2) for integer n0, n1, n2
    lhs = (R - np.eye(3)) @ r_center + t

    # Check if -lhs can be expressed as an integer linear combination of lattice basis vectors
    # If yes, then there exists a lattice translation that makes the atom invariant
    # n_vector contains the integer coefficients [n0, n1, n2]
    is_invariant, n_vector = is_lattice_vector(-lhs, lattice_basis, tolerance)

    return is_invariant,n_vector

# ==============================================================================
# STEP 7: Find neighboring atoms and partition into equivalence classes
# ==============================================================================

def generate_wyckoff_orbit(wyckoff_position, space_group_bilbao_cart, lattice_basis,
                          tolerance=1e-5, verbose=False):
    """
    Generate all symmetry-equivalent positions (orbit) from a single Wyckoff position.
    Applies all space group operations to a Wyckoff position and collects unique
     atomic positions within the unit cell. This generates the complete orbit of
    the Wyckoff position under the space group.

    For each operation [R|t], the transformation is:
        r' = R @ r + t
    where r is in fractional coordinates of the primitive cell.

     Positions that differ by a lattice vector are considered equivalent,
    so we reduce all positions to the range [0, 1) in fractional coordinates.


    :param wyckoff_position: dict from parsed_config['Wyckoff_positions']
                         Must contain 'fractional_coordinates' key
                         Example: {'position_name': 'C', 'atom_type': 'C',
                                  'fractional_coordinates': [0.33333333, 0.66666666, 0.0]}
    :param space_group_bilbao_cart: List of space group matrices in Cartesian coordinates
                                using Bilbao origin (shape: num_ops × 3 × 4)
    :param lattice_basis:  Primitive lattice basis vectors (3×3 array, each row is a basis vector)
                      expressed in Cartesian coordinates using Bilbao origin
    :param tolerance: Numerical tolerance for identifying duplicate positions (default: 1e-5)
    :param verbose: Whether to print debug information (default: False)
    :return: list of dicts: Each dict contains:
            - 'fractional_coordinates': [f0, f1, f2] in range [0, 1)
            - 'cartesian_coordinates': [x, y, z] in Cartesian coords (Bilbao origin)
            - 'operation_idx': which space group operation generated this position
            - 'position_name': inherited from input Wyckoff position
            - 'atom_type': inherited from input Wyckoff position
    """
    # Extract input position in fractional coordinates
    r_frac_input = np.array(wyckoff_position['fractional_coordinates'])
    position_name = wyckoff_position['position_name']
    atom_type = wyckoff_position['atom_type']

    # Convert lattice basis to proper array and get transformation matrices
    lattice_basis = np.array(lattice_basis)#rows are basis vectors
    lattice_matrix = np.column_stack(lattice_basis)  # Columns are basis vectors
    lattice_matrix_inv = np.linalg.inv(lattice_matrix)
    # Convert input fractional coordinates to Cartesian using Bilbao origin
    r_cart_input = frac_to_cartesian([0, 0, 0], r_frac_input, lattice_basis, origin_cart)

    # Store unique positions
    unique_positions = []
    unique_frac_coords = []  # For deduplication

    # Apply each space group operation
    for op_idx, operation in enumerate(space_group_bilbao_cart):
        # Extract rotation and translation
        R, t = get_rotation_translation(space_group_bilbao_cart, op_idx)
        # Apply symmetry operation in Cartesian coordinates
        # r_cart' = R @ r_cart + t
        r_cart_transformed = R @ r_cart_input + t

        # Convert back to fractional coordinates
        r_frac_transformed = lattice_matrix_inv @ r_cart_transformed

        # Wrap to [0, 1) to stay within unit cell
        r_frac_wrapped = r_frac_transformed % 1.0
        # Check if this position is already in our list (within tolerance)
        is_duplicate = False
        for existing_frac in unique_frac_coords:
            # Check if positions are equivalent (accounting for periodic boundary conditions)
            diff = r_frac_wrapped - existing_frac
            if np.linalg.norm(diff) < tolerance:
                is_duplicate = True
                break

        if not is_duplicate:
            # Add to unique positions
            unique_frac_coords.append(r_frac_wrapped)

            # Convert wrapped fractional back to Cartesian for output
            r_cart_final = frac_to_cartesian([0, 0, 0], r_frac_wrapped, lattice_basis, origin_cart)

            # Create position dictionary
            position_dict = {
                'fractional_coordinates': r_frac_wrapped.tolist(),
                'cartesian_coordinates': r_cart_final.tolist(),
                'operation_idx': op_idx,
                'position_name': position_name,
                'atom_type': atom_type
            }
            unique_positions.append(position_dict)

    return unique_positions



def generate_atoms_in_unit_cell(parsed_config,space_group_bilbao_cart, lattice_basis,origin_cart,repr_s, repr_p, repr_d, repr_f,
                          tolerance=1e-5):
    """
    Generates all atoms in the unit cell by expanding the Wyckoff positions defined
    in the configuration using the provided space group operations.
    Args:
        parsed_config: Dictionary containing configuration (Wyckoff positions, origin, etc.)
        space_group_bilbao_cart:  List of space group matrices in Cartesian coordinates
        lattice_basis:  3x3 array of lattice basis vectors (each row is a basis vector)
        repr_s, repr_p, repr_d, repr_f: Orbital representation matrices (numpy arrays)
        tolerance:  Numerical tolerance for coordinate comparisons

    Returns:
        list: A list of atomIndex objects representing all atoms in the unit cell [0,0,0]
    """

    unit_cell_atoms = []
    # Iterate over all Wyckoff positions defined in the configuration
    for wyckoff_pos in parsed_config['Wyckoff_positions']:
        # Generate the full orbit (all equivalent atoms in unit cell)\
        # This function applies space group operations to the Wyckoff generator
        orbit = generate_wyckoff_orbit(
            wyckoff_pos,
            space_group_bilbao_cart,
            lattice_basis,
            tolerance=tolerance
        )
        # Create atomIndex objects for each position in the orbit
        for index, pos_data in enumerate(orbit):
            atom = atomIndex(
                cell=[0, 0, 0],  # Always the home unit cell
                frac_coord=pos_data['fractional_coordinates'],
                position_name=pos_data['position_name'],
                basis=lattice_basis,
                origin_cart=origin_cart,
                parsed_config=parsed_config,
                repr_s_np=repr_s,
                repr_p_np=repr_p,
                repr_d_np=repr_d,
                repr_f_np=repr_f
            )
            wyckoff_instance_id_tmp=atom.position_name+str(index)
            atom.wyckoff_instance_id=wyckoff_instance_id_tmp
            unit_cell_atoms.append(atom)

    return unit_cell_atoms


unit_cell_atoms=generate_atoms_in_unit_cell(parsed_config, space_group_bilbao_cart, lattice_basis,origin_cart, repr_s, repr_p, repr_d, repr_f)

# ==============================================================================
# Define neighbor search parameters
# ==============================================================================
search_range=10 # Number of unit cells to search in each direction
               # Total search region: [-10, 10] × [-10, 10] for this 2d problem
               # Larger values find more distant neighbors but increase computation time

radius=1.05 * np.sqrt(3) # Cutoff distance in Cartesian coordinates
                         # Only atoms within this distance from center are considered neighbors
                         # Factor 1.05 provides small tolerance beyond sqrt(3) for numerical safety
                         # FIXME: search_range must be sufficiently large to include all atoms within radius
                         # TODO: may need an algorithm to deal with this in the next version of code


search_dim = 2  # Dimensionality of neighbor search
                # 1: Search along n0 only (1D chain)
                # 2: Search along n0 and n1 (2D layer) - appropriate for 2D materials like hBN
                # 3: Search along n0, n1, and n2 (3D bulk)


# ==============================================================================
# Find all neighbors for each atom in the unit cell
# ==============================================================================
# For each atom in the reference unit cell [0,0,0], find all neighboring atoms within
# the specified radius by searching through neighboring unit cells.
# This creates the hopping connectivity network for tight-binding calculations.


all_neighbors = {}  # Dictionary mapping unit cell atom index → list of neighbor atomIndex objects
                    # Key: integer index of center atom in unit_cell_atoms\
                    # Value: list of atomIndex objects representing all neighbors within radius
                    # Neighbors can be in different unit cells (n0, n1, n2)


for i, unit_atom in enumerate(unit_cell_atoms):
    # Find all neighbors within the specified radius for this center atom
    # The compute_dist function:
    # 1. Searches through neighboring unit cells within search_range
    # 2. Constructs atomIndex objects for atoms in those cells
    # 3. Filters by distance (keeps only atoms within radius)
    # 4. Returns sorted list by distance
    neighbors = compute_dist(
        center_atom=unit_atom,  # Center atom (in unit cell [0,0,0])
        unit_cell_atoms=unit_cell_atoms,  # Template atoms to replicate in neighboring cells
        search_range=search_range,  # How many cells to search in each direction
        radius=radius,  # Distance cutoff in Cartesian coordinates
        search_dim=search_dim  # Search dimensionality (here  is 2D)
    )
    # Store the neighbor list using the unit cell atom index as key
    # This creates a complete connectivity map: center_atom_idx --- [neighbor1, neighbor2, ...]
    all_neighbors[i] = neighbors
    # Print summary for each center atom
    print(f"Unit cell atom {i} ({unit_atom.atom_type}): found {len(neighbors)} neighbors within radius {radius}")



# ==============================================================================
# Find identity operation
# ==============================================================================
# Locate the identity operation E in the list of space group operations.
# The identity operation E = {identity matrix|0} is characterized by:
# - Rotation part: 3×3 identity matrix (no operation)
# - Translation part: zero vector (no translation)
#
# The identity operation index is crucial because:
# 1. It will be assigned to seed hoppings (root vertices in the constraint tree)
#    Seed hoppings are those containing identity operation
# 2. Root vertices in the vertex tree have hopping.operation_idx == identity_idx
#
# This index will be used throughout the code to:
# - Distinguish between seed hoppings and derived hoppings
# - Initialize root vertices in the constraint tree
# - Verify that orbital representations preserve identity (V[identity_idx] = identity matrix)

identity_idx = find_identity_operation(
    space_group_bilbao_cart,# List of space group operations in Cartesian coordinates
    tolerance=1e-5, # Numerical tolerance for comparing matrices to identity
    verbose=True# Print status message when identity is found
)
# print(f"identity_idx={identity_idx}")

# ==============================================================================
# print atom orbital representations for all unit cell atoms
# ==============================================================================
print("\n" + "=" * 80)
print("PRINTING ATOM ORBITAL REPRESENTATIONS")
print("=" * 80)
for i, atom in enumerate(unit_cell_atoms):
    print(f"\nUnit cell atom {i} ({atom.atom_type}):")
    print(f"  {atom}")
    print(f"  Orbitals: {atom.get_orbital_names()}")
    if atom.orbital_representations:
        print(f"  Number of operations: {len(atom.orbital_representations)}")
        V_identity = atom.get_representation_matrix(identity_idx)
        print(f" Orbital representation's identity matrix shape: {V_identity.shape}")
        print(f" Orbital representation's identity present: {np.allclose(V_identity, np.eye(V_identity.shape[0]))}")

print("\n" + "=" * 80)
print("ORBITAL REPRESENTATION VERIFICATION COMPLETE")
print("=" * 80)

# ==============================================================================
# Helper function for symmetry operations
# ==============================================================================

def bilbao_plus_translation(R,t,lattice_basis,n_vec,atom_cart):
    """
    Apply space group operation with lattice translation to an atom position.

    Computes the full symmetry transformation:
        r' = R @ r + t + n₀·a₀ + n₁·a₁ + n₂·a₂

    where:
        - R @ r is the rotation of the atom position
        - t is the fractional translation (origin shift) from the Bilbao space group operation
        - n₀·a₀ + n₁·a₁ + n₂·a₂ is a lattice vector translation

    This is the complete symmetry operation that includes the additional lattice
    translation needed to maintain the center atom invariance.

    :param R: 3×3 rotation matrix (in Cartesian coordinates, Bilbao origin)
    :param t: 3D translation vector (in Cartesian coordinates, Bilbao origin)
    :param lattice_basis: 3×3 array of primitive lattice basis vectors (each row is a basis vector)
                          expressed in Cartesian coordinates using Bilbao origin
    :param n_vec: Array [n₀, n₁, n₂] containing integer coefficients for lattice translation
    :param atom_cart: 3D Cartesian position of the atom (using Bilbao origin)
    :return: transformed_cart, 3D Cartesian position after applying the full symmetry operation
    """
    # Extract the three primitive lattice basis vectors (each row is one basis vector)
    a0=lattice_basis[0] # First primitive basis vector
    a1=lattice_basis[1]# Second primitive basis vector
    a2=lattice_basis[2]# Third primitive basis vector
    # Extract the integer coefficients for the lattice translation
    # These determine how many unit cells to shift along each basis direction
    n0,n1,n2=n_vec

    # Apply the complete symmetry transformation:
    # 1. R @ atom_cart: Apply rotation to the atom position
    # 2. + t: Add the Bilbao translation from the space group operation
    # 3. + n0*a0 + n1*a1 + n2*a2: Add the lattice vector translation
    #    This is the additional shift needed to preserve center atom invariance
    transformed_cart=R@atom_cart+t+n0*a0+n1*a1+n2*a2
    return transformed_cart



def get_next_for_center(center_atom, seed_atom, center_seed_distance, space_group_bilbao_cart,
                        operation_idx, parsed_config, tolerance=1e-5, verbose=False):
    """
     Apply a space group operation to a seed atom, conditioned on center atom invariance.
     This function implements a three-step validation process:
     1. Check if the center atom is invariant under the space group operation
       (usually with lattice translation). This determines the lattice shift n_vec.
     2. If invariant, apply the SAME operation (with the SAME n_vec) to the seed atom
        to generate an atom's Cartesian coordinate. This atom may be symmetry-equivalent
        to the seed atom.

     3. Verify that the transformed seed maintains the same distance from center.
     Physical Context:
     Given a seed hopping (center ← seed), this function applies a symmetry operation
     to generate a potentially equivalent hopping (center ← transformed_seed). The
     transformed position is only returned if it preserves the hopping distance,
     confirming it belongs to the same equivalence class, if the atom type also matches
     Args:
         center_atom: atomIndex object for the center atom (target of the hopping)
         seed_atom: atomIndex object for the seed neighbor atom (origin of seed hopping)
         center_seed_distance: Pre-computed distance from center to seed atom
                                (avoids redundant computation across operations)
        space_group_bilbao_cart: List of space group matrices in Cartesian coordinates
                                using Bilbao origin (shape: num_ops × 3 × 4)
        operation_idx: Index of the space group operation to apply
        parsed_config: Configuration dictionary containing lattice_basis
        tolerance: Numerical tolerance for invariance and distance checks (default: 1e-5)
        verbose: Whether to print debug information (default: False)
        Returns:
            numpy.ndarray or None:
                - Transformed Cartesian coordinates if:
                  (a) center is invariant under this operation, AND
                  (b) transformation preserves the center-seed distance
                - None otherwise (operation doesn't generate a valid equivalent hopping)

    """
    # ==============================================================================
    # Extract space group operation components
    # ==============================================================================
    # Get the rotation matrix R and translation vector b from the space group operation
    # The operation is represented as [R|b] in Cartesian coordinates
    R, b = get_rotation_translation(space_group_bilbao_cart, operation_idx)

    # ==============================================================================
    # Get lattice basis vectors
    # ==============================================================================
    # Extract the primitive lattice basis vectors from configuration
    # These are the fundamental translation vectors a0, a1, a2 of the crystal

    lattice_basis = np.array(parsed_config['lattice_basis'])
    # ==============================================================================
    # STEP 1: Check center atom invariance
    # ==============================================================================
    # Determine if the center atom is invariant under this space group operation,
    # usually with an additional lattice translation.
    ## Mathematical condition for invariance:
    #   R @ r_center + b + n_vec · [a0, a1, a2] = r_center
    # where n_vec = [n0, n1, n2] are integer coefficients for lattice translation
    #
    # This ensures that the symmetry operation preserves the hopping target -
    # the center atom remains at the same atomic site.
    #
    # Returns:
    #   is_invariant: True if center atom is invariant (with some lattice shift)
    #   n_vec: The required lattice translation [n0, n1, n2] for invariance
    is_invariant, n_vec = check_center_invariant(
        center_atom,
        operation_idx,
        space_group_bilbao_cart,
        lattice_basis,
        tolerance,
        verbose
    )
    # ==============================================================================
    # STEP 2: Apply operation to seed atom (only if center is invariant)
    # ==============================================================================
    if is_invariant:
        # Center atom is invariant under this operation (with lattice shift n_vec)
        # Now apply the SAME complete transformation to the seed atom:
        #   r_transformed = R @ r_seed + b + n_vec · [a0, a1, a2]
        #
        # KEY INSIGHT: We must use the SAME n_vec for the seed!
        # This ensures the hopping vector (r_center - r_seed) transforms consistently,
        # preserving the relative geometry of the hopping.
        ## The transformed position may correspond to an atom that is symmetry-equivalent
        # to the seed atom (same species, equivalent local environment).
        seed_cart_coord = seed_atom.cart_coord  # Original seed position
        # Apply the full symmetry transformation: R @ r + b + lattice_shift
        # This generates a Cartesian coordinate that may represent a symmetry-equivalent atom
        next_cart_coord = bilbao_plus_translation(R, b, lattice_basis, n_vec, seed_cart_coord)
        # ==============================================================================
        # STEP 3: Verify the transformation preserves hopping distance (isometry check)
        # ==============================================================================
        # Calculate the distance from the transformed position to center
        # For a valid symmetry operation (isometry), this must equal the original distance
        new_center_seed_dist = np.linalg.norm(next_cart_coord - center_atom.cart_coord, ord=2)
        # Check if the hopping distance is preserved
        # This verifies that: |r_center - (R @ r_seed + b + lattice_shift)| = |r_center - r_seed|
        dist_is_equal = (np.abs(new_center_seed_dist - center_seed_distance) < tolerance)
        # Only return the transformed position if distance is preserved
        # This ensures the generated coordinate represents a symmetry-equivalent atom
        if dist_is_equal == True:
            return (deepcopy(next_cart_coord), deepcopy(n_vec))
        else:
            # Distance not preserved - this indicates a numerical error or
            # inconsistency in the symmetry operation (shouldn't happen in practice)
            return None

    else:
        # ==============================================================================
        # Center atom is NOT invariant - operation invalid for this hopping
        # ==============================================================================
        # The center atom does not map to itself (even with lattice translations)
        # under this space group operation. Therefore, abandon this symmetry operation
        return None


def search_one_equivalent_atom(seed_atom,target_cart_coord, neighbor_atoms_copy, tolerance=1e-5, verbose=False):
    """
    Search for an atom in the neighbor_atoms_copy set whose Cartesian coordinate matches the target.
    This function is used to find which actual neighbor atom corresponds to a transformed
    position generated by a symmetry operation. If a match is found, it confirms that
    the symmetry operation maps the seed atom to an existing neighbor atom.
    Args:
        target_cart_coord: 3D Cartesian coordinate to search for (numpy array)
                            This is  the result of applying a symmetry operation
                            to a seed atom's position
        neighbor_atoms:  set of atomIndex objects representing all neighbors
                         of a center atom within some cutoff radius
        tolerance:  Numerical tolerance for coordinate comparison (default: 1e-5)
                    Two positions are considered identical if their Euclidean distance
                    is less than this tolerance
        verbose:  Whether to print debug information (default: False)
    Returns:
        atomIndex or None:
          - The matching neighbor atom if found (coordinate matches within tolerance)
            IMPORTANT: Returns a REFERENCE (not a copy) to the atomIndex object in neighbor_atoms
          - None if no match is found (transformed position doesn't correspond to
            any actual neighbor atom)
    """
    # Iterate through all neighbor atoms in the set
    for neighbor in neighbor_atoms_copy:
        # Compute Euclidean distance between target position and this neighbor's position
        distance = np.linalg.norm(target_cart_coord - neighbor.cart_coord, ord=2)
        # Check if the distance is within tolerance (positions match)
        if distance < tolerance and seed_atom.position_name == neighbor.position_name:
            # Return a REFERENCE (pointer in C sense, reference in C++ sense) to the matching neighbor atom
            # This is NOT a deep copy - it's the same object that exists in neighbor_atoms_copy
            # This allows the caller to use this reference to remove the atom from neighbor_atoms_copy
            return neighbor
    return None


def get_equivalent_sets_for_one_center_atom(center_atom_idx, unit_cell_atoms, all_neighbors,
                                                space_group_bilbao_cart, identity_idx,
                                                tolerance=1e-5, verbose=False):
    """
    Partition all neighbors of 1 center atom into equivalence classes based on symmetry.
    Each equivalence class contains center atom's neighbors related by space group operations.
    Algorithm:
    ---------
    1. Pop a seed atom from the remaining neighbors (arbitrary choice)
    2. Apply all space group operations to find symmetry-equivalent neighbors
    3. Group these equivalent neighbors together into one equivalence class
    4. Repeat until all neighbors are classified

    CRITICAL: Reference Handling
    ============================
    This function works with REFERENCES to atomIndex objects throughout:
    - neighbor_atoms_copy is a set of references to DEEP-COPIED atomIndex objects
    - seed_atom = set.pop() returns a reference to one of these copied objects
    - matched_neighbor from search is also a reference to one of these copied objects
    - equivalence_classes stores tuples containing references to these copied objects

    Why deep copy all_neighbors[center_atom_idx]?
    ---------------------------------------------
    We deep copy to DECOUPLE from the input:
    1. The input all_neighbors should remain unchanged (it may be used elsewhere)
    2. We destructively remove atoms from neighbor_atoms_copy as we classify them
    3. Deep copy creates NEW atomIndex objects (independent of the input)
    4. After deep copy:
        - all_neighbors[center_atom_idx] still has all its original atomIndex objects
        - neighbor_atoms_copy has completely separate atomIndex objects with same data
        - Modifying neighbor_atoms_copy has NO effect on all_neighbors
    Args:
        center_atom_idx: Index of the center atom in unit_cell_atoms
        unit_cell_atoms: List of all atomIndex objects in the unit cell
        all_neighbors: Dictionary mapping center atom index → list of neighbor atomIndex objects
        space_group_bilbao_cart: List of space group matrices in Cartesian coordinates
        identity_idx: Index of the identity operation
        tolerance: Numerical tolerance for comparisons (default: 1e-5)
        verbose: Whether to print debug information (default: False)
    Returns:
        List of equivalence classes, where each class is a list of tuples:
        (matched_neighbor, operation_idx, n_vec)
        where:
            - matched_neighbor: REFERENCE to deep-copied atomIndex object
            - operation_idx: Space group operation that maps seed → matched_neighbor
            - n_vec: Lattice translation vector [n₀, n₁, n₂] for this transformation
    """
    # ==============================================================================
    # Initialize working variables
    # ==============================================================================
    # Extract reference to center atom from unit cell
    # This is a REFERENCE (not copied) - center_atom points to the same object in unit_cell_atoms
    center_atom = unit_cell_atoms[center_atom_idx]
    # print(f"center_atom={center_atom}")
    # Create a working copy of neighbors as a set
    # IMPORTANT: Deep copy to DECOUPLE from input all_neighbors
    # ----------------------------------------------------------
    # Why deep copy?
    # - We will destructively remove atoms from neighbor_atoms_copy as we classify them
    # - We must NOT modify the input all_neighbors (caller may need it unchanged)
    # - Deep copy creates entirely NEW atomIndex objects (different memory addresses)
    #   with the same data as the originals
    #
    # Memory structure after deep copy:
    # - all_neighbors[center_atom_idx] = [obj_A, obj_B, obj_C, ...]  (original objects)
    # - neighbor_atoms_copy = {obj_A', obj_B', obj_C', ...}  (NEW copied objects)
    # - obj_A and obj_A' are DIFFERENT objects at DIFFERENT memory addresses
    # - obj_A and obj_A' have the SAME data (same coordinates, same element, etc.)
    # - Removing obj_A' from neighbor_atoms_copy does NOT affect all_neighbors
    #
    # Why set instead of list?
    # - O(1) removal with set.remove() vs O(n) with list.remove()
    # - No duplicates guaranteed
    # - Order doesn't matter (symmetry operations find all equivalents)
    neighbor_atoms_copy = set(deepcopy(all_neighbors[center_atom_idx]))
    # Store all equivalence classes (list of lists of tuples)
    equivalence_classes = []
    # Class ID counter (increments for each new equivalence class found)
    class_id = 0
    # ==============================================================================
    # Main loop: Partition neighbors into equivalence classes
    # ==============================================================================
    # Continue until all neighbors are classified into equivalence classes
    # Each iteration creates one equivalence class and removes its members from neighbor_atoms_copy
    while len(neighbor_atoms_copy) != 0:
        # ==============================================================================
        # STEP 1: Select seed atom for this equivalence class
        # ==============================================================================
        # Pop one seed atom from neighbor_atoms_copy
        # This will be the representative atom for this equivalence class
        #
        # CRITICAL: set.pop() returns a REFERENCE, not a copy
        # ------------------------------------------------
        # - set.pop() removes and returns a reference to an arbitrary element
        # - Order is implementation-dependent (hash table internals, not guaranteed)
        # - Returns a REFERENCE to one of the deep-copied atomIndex objects
        # - The atomIndex object is removed from the set but still exists in memory
        # - seed_atom now holds a reference to that object
        #
        # Example:
        # -------
        # Before: neighbor_atoms_copy = {obj_A', obj_B', obj_C'}
        # After:  seed_atom = obj_A' (reference to the copied object)
        #         neighbor_atoms_copy = {obj_B', obj_C'}
        #
        # Remember: obj_A' is a COPY (independent of the original obj_A in all_neighbors)
        #
        # The specific choice doesn't matter - symmetry operations will find all equivalent neighbors
        seed_atom = neighbor_atoms_copy.pop()
        # Pre-compute the distance from center to seed (used for all operations)
        # This distance must be preserved by symmetry operations (isometry)
        center_seed_distance = np.linalg.norm(center_atom.cart_coord - seed_atom.cart_coord, ord=2)
        # ==============================================================================
        # Initialize the current equivalence class
        # ==============================================================================
        # List of tuples: (neighbor_atom_reference, operation_idx, n_vec)
        current_equivalence_class = []
        # Add the seed atom itself with identity operation and zero lattice shift
        # The identity operation maps seed_atom to itself (by definition)
        # Tuple contains: (reference to seed_atom, identity_idx, zero vector)
        current_equivalence_class.append((seed_atom, identity_idx, np.array([0, 0, 0])))
        # ==============================================================================
        # STEP 2: Find all symmetry-equivalent neighbors
        # ==============================================================================
        # Iterate through all space group operations to find atoms equivalent to seed
        # Skip the identity operation since we already added the seed atom
        for operation_idx in range(len(space_group_bilbao_cart)):
            # Skip identity operation (already handled)
            if operation_idx == identity_idx:
                continue
            # Apply the space group operation to the seed atom
            # This generates a transformed position that may correspond to another neighbor
            # Returns (transformed_coord, n_vec) if valid, None otherwise
            result = get_next_for_center(
                center_atom=center_atom,
                seed_atom=seed_atom,
                center_seed_distance=center_seed_distance,
                space_group_bilbao_cart=space_group_bilbao_cart,
                operation_idx=operation_idx,
                parsed_config=parsed_config,
                tolerance=tolerance,
                verbose=verbose
            )
            # ==============================================================================
            # Process valid transformation results
            # ==============================================================================
            # If transformation is valid (center invariant, distance preserved)
            if result is not None:
                # Unpack the transformed coordinate and lattice shift vector
                # transformed_coord: 3D Cartesian position after applying symmetry operation
                # n_vec: Lattice translation [n₀, n₁, n₂] needed to preserve center invariance
                transformed_coord, n_vec = result
                # ==============================================================================
                # Search for matching neighbor in the remaining unclassified set
                # ==============================================================================
                # Search for this transformed position among the remaining neighbors
                # CRITICAL: matched_neighbor is a REFERENCE, not a copy
                # ---------------------------------------------------
                # search_one_equivalent_atom() returns:
                # - A REFERENCE to an atomIndex object in neighbor_atoms_copy if match found
                # - None if no match found
                #
                # This reference is ESSENTIAL for set.remove() to work:
                # - Python's set.remove() uses object identity (memory address)
                # - We need the EXACT SAME object reference that's in the set
                # - A copy wouldn't work (different object, different identity)
                #
                # Remember: matched_neighbor references a COPIED atomIndex object (obj_X')
                # NOT an original from all_neighbors (obj_X)
                matched_neighbor = search_one_equivalent_atom(
                    seed_atom,
                    target_cart_coord=transformed_coord,
                    neighbor_atoms_copy=neighbor_atoms_copy,
                    tolerance=tolerance,
                    verbose=verbose
                )
                # ==============================================================================
                # Add matched neighbor to equivalence class
                # ==============================================================================
                # If we found a matching neighbor in the remaining set
                if matched_neighbor is not None:
                    # Add to current equivalence class
                    # Store tuple: (reference to matched_neighbor, operation_idx, copy of n_vec)
                    # - matched_neighbor: REFERENCE to a deep-copied atomIndex object (from neighbor_atoms_copy)
                    # - operation_idx: Which space group operation maps seed → matched_neighbor
                    # - deepcopy(n_vec): Copy of lattice translation vector (n_vec is numpy array, mutable)
                    current_equivalence_class.append((matched_neighbor, operation_idx, deepcopy(n_vec)))
                    # Remove from the working set (it's now classified)
                    # CRITICAL: This only works because matched_neighbor is a REFERENCE
                    # ----------------------------------------------------------------
                    # set.remove() searches for object by identity (memory address)
                    # - matched_neighbor points to the exact same object in neighbor_atoms_copy
                    # - Python finds the object by comparing memory addresses (fast, O(1))
                    # - If matched_neighbor were a copy, remove() would raise KeyError
                    #
                    # After removal:
                    # - The atomIndex object still exists in memory (referenced by matched_neighbor
                    #   and by the tuple in current_equivalence_class)
                    # - It's just no longer in the neighbor_atoms_copy set
                    # - The original object in all_neighbors is completely unaffected
                    neighbor_atoms_copy.remove(matched_neighbor)
        # ==============================================================================
        # Complete this equivalence class
        # ==============================================================================
        # Add the completed equivalence class to the list
        # equivalence_classes is a list of lists of tuples
        # Each tuple contains: (reference to deep-copied atomIndex, operation_idx, n_vec)
        equivalence_classes.append(current_equivalence_class)
        # Increment class ID for next equivalence class
        class_id += 1

    # ==============================================================================
    # Return results
    # ==============================================================================
    return equivalence_classes


lattice_basis = np.array(parsed_config['lattice_basis'])
type_linear="linear"
type_hermitian="hermitian"
atom0=unit_cell_atoms[0]
print(f"atom0: {atom0}")
print(f"identity_idx={identity_idx}")

equivalence_classes_center_atom_0 = get_equivalent_sets_for_one_center_atom(0, unit_cell_atoms, all_neighbors,
                                                                                    space_group_bilbao_cart,
                                                                                    identity_idx)


ind=3
print(len(equivalence_classes_center_atom_0[ind]))
print(equivalence_classes_center_atom_0[ind])