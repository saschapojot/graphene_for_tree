import subprocess
import sys
import json
import numpy as np
from datetime import datetime
from copy import deepcopy
from pathlib import Path
import sympy as sp
import pickle
import base64
sp.init_printing(use_unicode=False, wrap_line=False)

#self defined
from classes.class_defs import frac_to_cartesian, atomIndex, hopping, vertex, T_tilde_total

#this script computes for general

# ==============================================================================
# STEP 1: Validate command line arguments
# ==============================================================================

argErrCode = 20
save_err_code=30
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




# Combine parsed_config and space_group_representations
combined_input = {
    "parsed_config": parsed_config,
    "space_group_representations": space_group_representations
}


# Convert to JSON for subprocess
combined_input_json = json.dumps(combined_input)

#run checking space group compatibility
compatibility_result= subprocess.run(
    ["python3", "./symmetry/check_space_group_compatible.py"],
    input=combined_input_json,
    capture_output=True,
    text=True
)
# Check ifchecking space group compatibility succeeded
if compatibility_result.returncode != 0:
    print("Compatibility check failed!")
    print(f"Return code: {compatibility_result.returncode}")
    print("Error output:")
    print(compatibility_result.stderr)
    exit(compatibility_result.returncode)
else:
    # This else block was added to print the captured stdout
    print(compatibility_result.stdout)