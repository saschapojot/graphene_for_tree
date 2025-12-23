import sys
import glob
import re
import json
import numpy as np

# ==============================================================================
# Sanity check script for tight-binding configuration files
# ==============================================================================
# This script validates the parsed configuration data from .conf files
# It checks for:
# - Valid matrix properties (determinant, condition number)
# - Valid Wyckoff position type references
# - Duplicate atomic positions after lattice reduction

# Exit codes for different error conditions
jsonErr = 4                      # JSON parsing error
valErr = 5                       # Value validation error
matrix_not_exist_error = 6       # Required matrix field missing
matrix_cond_error = 7            # Matrix condition number or determinant error
atom_position_error = 8          # Atom position reference error
duplicate_position_error = 9     # Duplicate atomic positions found


# ==============================================================================
# STEP 1: Read and parse JSON input from stdin
# ==============================================================================
try:
    config_json = sys.stdin.read()
    parsed_config = json.loads(config_json)

except json.JSONDecodeError as e:
    print(f"Error parsing JSON input: {e}", file=sys.stderr)
    exit(jsonErr)


# ==============================================================================
# STEP 2: Define matrix validation function
# ==============================================================================
def check_matrix_condition(matrix, matrix_name="Matrix", det_threshold=1e-12, cond_threshold=1e12):
    """
    Check if a matrix is well-conditioned and non-degenerate

    A valid matrix should:
    1. Be square (n×n)
    2. Have non-zero determinant (non-degenerate)
    3. Have reasonable condition number (well-conditioned)

    :param matrix: 2D list or numpy array representing the matrix
    :param matrix_name: Name of the matrix for error messages
    :param det_threshold: Minimum absolute determinant value (default: 1e-12)
    :param cond_threshold: Maximum condition number (default: 1e12)
    :return: tuple: (is_valid, error_message)
               is_valid: True if matrix passes all checks
               error_message: None if valid, error string if invalid
    """
    try:
        # Convert to numpy array if it's a list
        if isinstance(matrix, list):
            np_matrix = np.array(matrix)
        else:
            np_matrix = matrix

        # Check if it's a square matrix (required for basis vectors)
        if np_matrix.shape[0] != np_matrix.shape[1]:
            return False, f"{matrix_name} is not square: shape {np_matrix.shape}"

        # Check determinant (non-degenerate test)
        # A zero determinant means the vectors are linearly dependent
        det = np.linalg.det(np_matrix)
        if abs(det) < det_threshold:
            return False, f"{matrix_name} is degenerate (determinant ≈ 0): det = {det:.2e}"

        # Check condition number (ill-conditioning test)
        # High condition number indicates numerical instability
        cond_num = np.linalg.cond(np_matrix)
        if cond_num > cond_threshold:
            return False, f"{matrix_name} is ill-conditioned: condition number = {cond_num:.2e}"

        return True, None

    except Exception as e:
        return False, f"Error analyzing {matrix_name}: {str(e)}"


# ==============================================================================
# STEP 3: Check for required matrix fields
# ==============================================================================
# Verify that lattice_basis exists and is not empty
if 'lattice_basis' not in parsed_config or not parsed_config['lattice_basis']:
    print("Error: Missing or empty required field 'lattice_basis'", file=sys.stderr)
    exit(matrix_not_exist_error)

# Verify that space_group_basis exists and is not empty
if 'space_group_basis' not in parsed_config or not parsed_config['space_group_basis']:
    print("Error: Missing or empty required field 'space_group_basis'", file=sys.stderr)
    exit(matrix_not_exist_error)


# ==============================================================================
# STEP 4: Validate matrix conditions
# ==============================================================================
# Check lattice basis matrix properties
is_valid, error_msg = check_matrix_condition(parsed_config['lattice_basis'], "Lattice basis")
if not is_valid:
    print(f"Error: {error_msg}", file=sys.stderr)
    exit(matrix_cond_error)

# Check space group basis matrix properties
is_valid, error_msg = check_matrix_condition(parsed_config['space_group_basis'], "Space group basis")
if not is_valid:
    print(f"Error: {error_msg}", file=sys.stderr)
    exit(matrix_cond_error)


# ==============================================================================
# STEP 5: Define Wyckoff position validation function
# ==============================================================================
def check_Wyckoff_positions(parsed_config):
    """
    Check that every Wyckoff position refers to a valid Wyckoff position type.

    Verifies that:
    - Wyckoff_position_types exists.
    - Wyckoff_positions exists.
    - Every position in Wyckoff_positions has an 'atom_type' that exists as a key
      in Wyckoff_position_types.

    :param parsed_config: Parsed configuration dictionary
    :return: tuple: (is_valid, error_message)
    """
    # Extract definitions and position data using new keys
    # 'atom_types' is replaced by 'Wyckoff_position_types'
    wyckoff_types = parsed_config.get('Wyckoff_position_types', {})
    wyckoff_positions = parsed_config.get('Wyckoff_positions', [])

    # Basic validation
    if not wyckoff_types:
        return False, "No Wyckoff_position_types defined"

    if not wyckoff_positions:
        return False, "No Wyckoff_positions defined"

    # Verify referential integrity
    for i, position in enumerate(wyckoff_positions):
        # Get the references from the position
        p_name = position.get('position_name')

        # Ensure position_name exists
        if not p_name:
            return False, f"Position at index {i} is missing the 'position_name' field"

        # Check if the position_name (e.g., 'C0') is defined in the types dictionary
        if p_name not in wyckoff_types:
            return False, f"Position '{p_name}' is not defined in Wyckoff_position_types"

    return True, None


# ==============================================================================
# STEP 6: Define duplicate position detection function
# ==============================================================================
def check_duplicate_positions(parsed_config, tolerance=1e-6):
    """
    Check for duplicate atomic positions after reducing by lattice vectors

    Atomic positions that differ by lattice vectors are physically identical.
    This function reduces all positions modulo the lattice and checks for duplicates.

    :param parsed_config: Parsed configuration dictionary
    :param tolerance: Distance threshold for considering positions duplicate (default: 1e-6)
    :return: tuple: (is_valid, error_message)
    """
    # Extract lattice basis and atom positions
    lattice_basis = parsed_config.get('lattice_basis')
    wyckoff_positions = parsed_config.get('Wyckoff_positions', [])

    # Basic validation
    if not lattice_basis:
        return False, "Lattice basis not found"
    if not wyckoff_positions:
        return True, None  # No positions to check

    try:
        # Convert lattice basis to numpy array
        lattice_matrix = np.array(lattice_basis)

        # Store reduced positions for comparison
        reduced_positions = []
        position_info = []  # For error reporting

        # Process each atom position
        for i, position in enumerate(wyckoff_positions):
            # Determine display name for error messages
            position_name = position.get('position_name')
            atom_type = position.get('atom_type')

            if position_name:
                display_name = position_name
            elif atom_type:
                display_name = atom_type
            else:
                display_name = f'atom_{i}'

            # Get fractional coordinates
            coords = position.get('fractional_coordinates')

            if not coords or len(coords) != 3:
                return False, f"Invalid fractional_coordinates for {display_name} at index {i}"

            # Reduce coordinates to [0, 1) in fractional coordinates
            # This handles periodicity: [0.1, 0.2, 0.3] and [1.1, 1.2, 1.3] are equivalent
            coord_array = np.array(coords, dtype=float)
            reduced_coord = coord_array % 1.0

            # Handle numerical precision issues near boundaries
            # e.g., 0.9999999 should be treated as 0.0
            reduced_coord = np.where(reduced_coord > 1.0 - tolerance, 0.0, reduced_coord)

            reduced_positions.append(reduced_coord)
            position_info.append((display_name, i, coords))

        # Check for duplicates by comparing all pairs
        for i in range(len(reduced_positions)):
            for j in range(i + 1, len(reduced_positions)):
                pos1 = reduced_positions[i]
                pos2 = reduced_positions[j]

                # Calculate distance considering periodic boundary conditions
                diff = pos1 - pos2

                # Handle wraparound: positions near 0 and 1 should be compared correctly
                # e.g., 0.999 and 0.001 should have distance ≈ 0.002, not 0.998
                diff = np.where(diff > 0.5, diff - 1.0, diff)
                diff = np.where(diff < -0.5, diff + 1.0, diff)

                distance = np.linalg.norm(diff)

                # If distance is below threshold, positions are duplicates
                if distance < tolerance:
                    name1, idx1, coords1 = position_info[i]
                    name2, idx2, coords2 = position_info[j]
                    return False, (f"Duplicate positions found: {name1} at {coords1} "
                                   f"and {name2} at {coords2} are equivalent after "
                                   f"lattice reduction (distance: {distance:.2e})")

        return True, None

    except Exception as e:
        return False, f"Error checking duplicate positions: {str(e)}"


# ==============================================================================
# STEP 7: Validate wyckoff positions match definitions
# ==============================================================================
is_valid, error_msg = check_Wyckoff_positions(parsed_config)
if not is_valid:
    print(f"Error: {error_msg}", file=sys.stderr)
    exit(atom_position_error)


# ==============================================================================
# STEP 8: Check for duplicate positions
# ==============================================================================
is_valid, error_msg = check_duplicate_positions(parsed_config)
if not is_valid:
    print(f"Error: {error_msg}", file=sys.stderr)
    exit(duplicate_position_error)


# ==============================================================================
# STEP 9: All checks passed - output success message
# ==============================================================================
print("SUCCESS: All sanity checks passed!", file=sys.stdout)