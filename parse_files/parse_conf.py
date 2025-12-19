import re
import sys
import json
import os

# ==============================================================================
# Configuration parser for tight-binding model input files
# ==============================================================================
# This script parses .conf files containing lattice, atom, and orbital information


# Exit codes for different error types
fmtErrStr = "format error: "
formatErrCode = 1        # Format/syntax errors in conf file
valueMissingCode = 2     # Required values are missing
paramErrCode = 3         # Wrong command-line parameters
fileNotExistErrCode = 4  # Configuration file doesn't exist


# ==============================================================================
# STEP 0: Validate command-line arguments
# ==============================================================================
if len(sys.argv) != 2:
    print("wrong number of arguments.", file=sys.stderr)
    print("usage: python parse_conf.py /path/to/xxx.conf", file=sys.stderr)
    exit(paramErrCode)

conf_file = sys.argv[1]

# Check if configuration file exists
if not os.path.exists(conf_file):
    print(f"file not found: {conf_file}", file=sys.stderr)
    exit(fileNotExistErrCode)


# ==============================================================================
# STEP 1: Define regex patterns for parsing
# ==============================================================================
# General key=value pattern
key_value_pattern = r'^([^=\s]+)\s*=\s*([^=]+)\s*$'

# Pattern for floating point numbers (including scientific notation)
float_pattern = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"

# Pattern for atom type definitions: AtomSymbol = orbital1, orbital2, ...
# Example: O=2px,2py,2pz
# Modified to remove the count and semicolon requirement
atom_orbital_pattern = r'^([A-Za-z]+\d*)\s*=\s*([1-7](?:s|px|py|pz|dxy|dxz|dyz|dx2-y2|dz2|fxyz|fx3-3xy2|f3x2y-y3|fxz2|fyz2|fx2-y2z|fz3)(?:\s*,\s*[1-7](?:s|px|py|pz|dxy|dxz|dyz|dx2-y2|dz2|fxyz|fx3-3xy2|f3x2y-y3|fxz2|fyz2|fx2-y2z|fz3))*)\s*$'

# Pattern for system name
name_pattern = r'^name\s*=\s*([a-zA-Z0-9_-]+)\s*$'

# Pattern for dimensionality (2 or 3)
dim_pattern = r"^dim\s*=\s*(\d+)\s*$"

# Pattern for number of neighbor cells to consider
neighbors_pattern = r"^neighbors\s*=\s*(\d+)\s*$"

# Pattern for number of Wyckoff position types
Wyckoff_type_num_pattern = r"^Wyckoff_type_num\s*=\s*(\d+)\s*$"

# Pattern for atom position coefficients (fractional coordinates)
# Example: O1_position_coefs = 0.5, 0.5, 0.0
atom_position_pattern_3d = rf'^([a-zA-Z]+\d*)_position_coefs\s*=\s*({float_pattern})\s*,\s*({float_pattern})\s*,\s*({float_pattern})\s*$'

# Pattern for lattice basis vectors: v1 ; v2 ; v3
# Example: lattice_basis = 1.0,0.0,0.0 ; 0.0,1.0,0.0 ; 0.0,0.0,1.0
lattice_basis_pattern = rf'^lattice_basis\s*=\s*({float_pattern}\s*,\s*{float_pattern}\s*,\s*{float_pattern})(?:\s*;\s*({float_pattern}\s*,\s*{float_pattern}\s*,\s*{float_pattern})){{2}}\s*$'

# Pattern for space group number (1-230)
space_group_pattern = r"^space_group\s*=\s*(\d+)\s*$"

# Pattern for space group origin in fractional coordinates
space_group_origin_pattern = rf"^space_group_origin\s*=\s*({float_pattern})\s*,\s*({float_pattern})\s*,\s*({float_pattern})\s*$"

# Pattern for space group basis vectors
space_group_basis_pattern = rf"^space_group_basis\s*=\s*({float_pattern}\s*,\s*{float_pattern}\s*,\s*{float_pattern})(?:\s*;\s*({float_pattern}\s*,\s*{float_pattern}\s*,\s*{float_pattern})){{2}}\s*$"

# Pattern for spin flag (true/false)
spin_pattern = r'^spin\s*=\s*((?i:true|false))\s*$'

# Pattern for lattice type (primitive or conventional)
lattice_type_pattern = r'^lattice_type\s*=\s*((?i:primitive|conventional))\s*$'


# ==============================================================================
# STEP 2: Define helper function to clean file contents
# ==============================================================================
def removeCommentsAndEmptyLines(file):
    """
    Remove comments and empty lines from configuration file

    Comments start with # and continue to end of line
    Empty lines (or lines with only whitespace) are removed

    :param file: conf file path
    :return: list of cleaned lines (comments and empty lines removed)
    """
    with open(file, "r") as fptr:
        lines = fptr.readlines()

    linesToReturn = []
    for oneLine in lines:
        # Remove comments (everything after #) and strip whitespace
        oneLine = re.sub(r'#.*$', '', oneLine).strip()

        # Only add non-empty lines
        if oneLine:
            linesToReturn.append(oneLine)

    return linesToReturn


# ==============================================================================
# STEP 3: Define main parsing function
# ==============================================================================
def parseConfContents(file):
    """
    Parse configuration file contents into structured dictionary

    Extracts:
    - System parameters (name, dimensions, spin, neighbors)
    - Lattice information (basis vectors, type)
    - Space group information (number, origin, basis)
    - Atom type on each Wyckoff position and their orbitals
    - Atom positions in fractional coordinates

    :param file: conf file path
    :return: dictionary containing parsed configuration
    """
    # Get cleaned lines from file
    linesWithCommentsRemoved = removeCommentsAndEmptyLines(file)

    # Initialize result dictionary with all expected fields
    config = {
        'name': '',                    # System name
        'dim': '',                     # Dimensionality (2 or 3)
        'spin': '',                    # Spin consideration (true/false)
        'neighbors': '',               # Number of neighbor cells to consider
        'Wyckoff_type_num': '',          # Total number of Wyckoff position types
        'lattice_type': '',           # Lattice type (primitive/conventional)
        'lattice_basis': '',          # Lattice basis vectors (3x3 matrix)
        'space_group': '',            # Space group number
        'space_group_origin': '',     # Space group origin (fractional coords)
        'space_group_basis': '',      # Space group basis vectors
        'Wyckoff_position_types': {}, # Dictionary: Wyckoff_position_types -> {orbitals}
        'Wyckoff_positions': []          # List of atom positions with types
    }

    # Parse each line
    for oneLine in linesWithCommentsRemoved:
        # Check if line matches key=value format
        matchLine = re.match(key_value_pattern, oneLine)

        if matchLine:
            # ==========================================
            # Parse system name
            # ==========================================
            name_match = re.match(name_pattern, oneLine)
            if name_match:
                config['name'] = name_match.group(1)
                continue

            # ==========================================
            # Parse dimensionality (2D or 3D)
            # ==========================================
            dim_match = re.match(dim_pattern, oneLine)
            if dim_match:
                config['dim'] = int(dim_match.group(1))
                continue

            # ==========================================
            # Parse spin flag
            # ==========================================
            spin_match = re.match(spin_pattern, oneLine)
            if spin_match:
                config['spin'] = spin_match.group(1)
                continue

            # ==========================================
            # Parse number of neighbor cells
            # ==========================================
            match_neighbors = re.match(neighbors_pattern, oneLine)
            if match_neighbors:
                config['neighbors'] = int(match_neighbors.group(1))
                continue

            # ==========================================
            # Parse number of Wyckoff types
            # ==========================================
            match_wyckoff_num = re.match(Wyckoff_type_num_pattern, oneLine)
            if match_wyckoff_num:
                config['Wyckoff_type_num'] = int(match_wyckoff_num.group(1))
                continue

            # ==========================================
            # Parse lattice type
            # ==========================================
            match_lattice_type = re.match(lattice_type_pattern, oneLine)
            if match_lattice_type:
                config['lattice_type'] = match_lattice_type.group(1)
                continue

            # ==========================================
            # Parse lattice basis vectors
            # Format: v1x,v1y,v1z ; v2x,v2y,v2z ; v3x,v3y,v3z
            # ==========================================
            match_lattice_basis = re.match(lattice_basis_pattern, oneLine)
            if match_lattice_basis:
                # Extract the full lattice basis value after the = sign
                full_value = oneLine.split('=')[1].strip()

                # Split into 3 vectors separated by semicolons
                vectors = []
                for vector in full_value.split(';'):
                    # Parse x,y,z coordinates for each vector
                    coords = [float(x.strip()) for x in vector.strip().split(',')]
                    vectors.append(coords)

                config['lattice_basis'] = vectors
                continue

            # ==========================================
            # Parse space group number
            # ==========================================
            match_space_group = re.match(space_group_pattern, oneLine)
            if match_space_group:
                config['space_group'] = int(match_space_group.group(1))
                continue

            # ==========================================
            # Parse space group origin (fractional coordinates)
            # ==========================================
            match_space_group_origin = re.match(space_group_origin_pattern, oneLine)
            if match_space_group_origin:
                x_coord = float(match_space_group_origin.group(1))
                y_coord = float(match_space_group_origin.group(2))
                z_coord = float(match_space_group_origin.group(3))
                config['space_group_origin'] = [x_coord, y_coord, z_coord]
                continue

            # ==========================================
            # Parse space group basis vectors
            # ==========================================
            match_space_group_basis = re.match(space_group_basis_pattern, oneLine)
            if match_space_group_basis:
                full_value = oneLine.split('=')[1].strip()
                vectors = []
                for vector in full_value.split(';'):
                    coords = [float(x.strip()) for x in vector.strip().split(',')]
                    vectors.append(coords)
                config['space_group_basis'] = vectors
                continue

            # ==========================================
            # Parse atom type (Wyckoff position) definitions
            # Format: AtomSymbol = orbital1, orbital2, ...
            # Example: B = 2pz, 2s
            # ==========================================
            atom_match = re.match(atom_orbital_pattern, oneLine)
            if atom_match:
                atom_type = atom_match.group(1)      # Atom symbol (B, N, O, etc.)

                # Group 2 is now the orbital string (count was removed from regex)
                orbitals = [o.strip() for o in atom_match.group(2).split(',')]

                # Store atom type information
                config['Wyckoff_position_types'][atom_type] = {

                    'orbitals': orbitals
                }
                continue

            # ==========================================
            # Parse atom position coefficients (fractional coordinates)
            # Format: AtomName_position_coefs = x, y, z
            # Example: O1_position_coefs = 0.5, 0.5, 0.0
            # ==========================================
            coefs_match = re.match(atom_position_pattern_3d, oneLine)
            if coefs_match:
                position_name = coefs_match.group(1)  # Position name (A, B, O1, O2, etc.)
                x_coord = float(coefs_match.group(2))
                y_coord = float(coefs_match.group(3))
                z_coord = float(coefs_match.group(4))

                # Determine the base atom type by removing trailing numbers
                # Example: O1 -> O, O2 -> O, B -> B
                base_atom_type = re.sub(r'\d+$', '', position_name)

                # Store position information
                position_info = {
                    'position_name': position_name,                    # Full name (O1, O2, etc.)
                    'atom_type': base_atom_type,                       # Base type (O, B, etc.)
                    'fractional_coordinates': [x_coord, y_coord, z_coord]  # Fractional coords
                }
                config['Wyckoff_positions'].append(position_info)
                continue

            # If no pattern matched, log unrecognized line
            print(f"Unrecognized key-value line: {oneLine}", file=sys.stderr)

        else:
            # Line doesn't match key=value format
            print("line: " + oneLine + " is discarded.", file=sys.stderr)

    return config


# ==============================================================================
# STEP 4: Parse configuration and output as JSON
# ==============================================================================
# Parse the configuration file
parsed_config = parseConfContents(conf_file)

# Output the parsed configuration as JSON to stdout
# This allows the data to be piped to other scripts
print(json.dumps(parsed_config, indent=2), file=sys.stdout)