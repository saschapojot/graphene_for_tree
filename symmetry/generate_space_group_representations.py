import numpy as np
import sys
import json
import re
import copy

# ==============================================================================
# Space group representation computation script
# ==============================================================================
# Original file: /home/adada/Documents/pyCode/TB/cd/SymGroup.py
# This script computes space group representations for atomic orbitals
# It transforms space group operations between different coordinate systems:
# - Bilbao basis (standard crystallographic basis)
# - Cartesian basis (x, y, z coordinates)
# - Primitive cell basis (lattice vectors as basis)
#
# It also computes how symmetry operations act on atomic orbitals (s, p, d, f)

# Exit codes for different error conditions
json_err_code = 4   # JSON parsing error
key_err_code = 5    # Required key missing from configuration
val_err_code = 6    # Invalid value in configuration


# ==============================================================================
# STEP 1: Read and parse JSON input from stdin
# ==============================================================================
try:
    config_json = sys.stdin.read()
    parsed_config = json.loads(config_json)

except json.JSONDecodeError as e:
    print(f"Error parsing JSON input: {e}", file=sys.stderr)
    exit(json_err_code)


# ==============================================================================
# STEP 2: Extract space group configuration data
# ==============================================================================
# Note: All operations assume primitive cell basis unless otherwise specified

try:
    # Primitive cell lattice basis vectors (3x3 matrix)
    # Each row is a lattice vector in Cartesian coordinates
    lattice_basis_primitive = parsed_config['lattice_basis']
    lattice_basis_primitive = np.array(lattice_basis_primitive)

    # Space group number (1-230 for 3D crystals)
    space_group = parsed_config['space_group']

    # Origin of the space group in fractional coordinates
    # This is the Bilbao origin choice
    space_group_origin = parsed_config['space_group_origin']
    space_group_origin = np.array(space_group_origin)

    # Basis vectors for the space group (Bilbao convention)
    # Each row is a basis vector in Cartesian coordinates
    space_group_basis = parsed_config['space_group_basis']
    space_group_basis = np.array(space_group_basis)
    # Convert Bilbao origin to Cartesian coordinates
    space_group_origin_cart = space_group_origin @ space_group_basis

    space_group_origin_frac_primitive = space_group_origin_cart @ np.linalg.inv(lattice_basis_primitive.T)




except KeyError as e:
    print(f"Error: Required key {e} not found in configuration", file=sys.stderr)
    exit(key_err_code)
except ValueError as e:
    print(f"Error with configuration data: {e}", file=sys.stderr)
    exit(val_err_code)


# ==============================================================================
# STEP 3: Define utility function to clean file contents
# ==============================================================================
def removeCommentsAndEmptyLines(file):
    """
    Remove comments and empty lines from file

    Comments start with # and continue to end of line

    :param file: File path
    :return: List of cleaned lines (comments and empty lines removed)
    """
    with open(file, "r") as fptr:
        lines = fptr.readlines()

    linesToReturn = []
    for oneLine in lines:
        # Remove comments (everything after #) and strip whitespace
        oneLine = re.sub(r'#.*$', '', oneLine).strip()
        if oneLine:  # Only add non-empty lines
            linesToReturn.append(oneLine)

    return linesToReturn


# ==============================================================================
# STEP 4: Define function to read space group matrices from file
# ==============================================================================
def read_space_group(in_space_group_file, space_group_num):
    """
    Read space group symmetry operations from database file

    The file contains space group operations in affine matrix form:
    [ R | t ] where R is 3x3 rotation/reflection and t is 3x1 translation
    Stored as 3x4 matrices: [R11 R12 R13 t1]
                           [R21 R22 R23 t2]
                           [R31 R32 R33 t3]

    :param in_space_group_file: File containing matrices of all space groups
    :param space_group_num: Space group number (1-230)
    :return: Space group matrices (affine) for space_group_num, shape (num_ops, 3, 4)
    """
    contents = removeCommentsAndEmptyLines(in_space_group_file)

    # Regex patterns for parsing
    # Space group header: "_<space_group_num>_ <num_matrices>"
    space_group_pattern = r'_(\d+)_\s+(\d+)'
    # Matrix element: integer or fraction like "1/2", "-1/2"
    matrix_elem_pattern = r'([+-]?\d+(?:/\d+)?)'

    for line_num in range(len(contents)):
        match_space_group = re.match(space_group_pattern, contents[line_num])
        if match_space_group:
            # Found a space group header
            sgn = int(match_space_group.group(1))  # Space group number

            if sgn == space_group_num:  # Found the desired space group
                num_matrices = int(match_space_group.group(2))  # Number of symmetry operations

                # Initialize array to store space group matrices
                # First 3 columns: linear part (rotation/reflection)
                # Last column: translation part
                space_group_matrices = np.zeros((num_matrices, 3, 4))

                # Read the matrices following the space group header
                for matrix_idx in range(num_matrices):
                    matrix_line = contents[line_num + matrix_idx + 1]
                    elements = re.findall(matrix_elem_pattern, matrix_line)

                    if len(elements) != 12:
                        raise ValueError(f"Expected 12 elements, got {len(elements)} in line: {matrix_line}")

                    # Parse 12 elements (3x4 matrix flattened row-wise)
                    matrix_elements = []
                    for one_elem in elements:
                        if "/" in one_elem:
                            # Handle fractions like "1/2", "-1/2"
                            numerator, denominator = one_elem.split("/")
                            matrix_elements.append(float(numerator) / float(denominator))
                        else:
                            # Handle integers like "1", "-1", "+1"
                            matrix_elements.append(float(one_elem))

                    # Reshape to 3x4 and store
                    space_group_matrices[matrix_idx] = np.array(matrix_elements).reshape((3, 4))

                return space_group_matrices

    # If space group not found after scanning entire file
    raise ValueError(f"Space group {space_group_num} not found in {in_space_group_file}")


# ==============================================================================
# STEP 5: Define coordinate transformation functions
# ==============================================================================
def space_group_to_cartesian_basis(space_group_matrices, space_group_basis):
    """
    Transform space group operations from Bilbao basis to Cartesian basis

    Original function: GetSymXyz(SymLvSG, LvSG) in cd/SymGroup.py

    Transformation formula for affine matrix [R|t]:
    - R_cart = A^T @ R_bilbao @ (A^T)^(-1)
    - t_cart = A^T @ t_bilbao
    where A is the space group basis matrix (rows are basis vectors)

    :param space_group_matrices: Space group operators (affine) in Bilbao basis
    :param space_group_basis: The basis of space_group_matrices (rows are basis vectors in Cartesian coords)
    :return: Space group operators under Cartesian coordinates, shape (num_ops, 3, 4)
    """
    A = space_group_basis
    AT = space_group_basis.T      # Transpose for column-vector representation
    AT_inv = np.linalg.inv(AT)

    num_operators = len(space_group_matrices)

    space_group_matrices_cartesian = np.zeros((num_operators, 3, 4), dtype=float)
    for j in range(num_operators):
        # Transform rotation/reflection part
        space_group_matrices_cartesian[j, :, 0:3] = AT @ space_group_matrices[j, :, 0:3] @ AT_inv
        # Transform translation part
        space_group_matrices_cartesian[j, :, 3] = AT @ space_group_matrices[j, :, 3]

    return space_group_matrices_cartesian


def space_group_to_primitive_cell_basis(space_group_matrices_cartesian, lattice_basis_primitive):
    """
    Transform space group operations from Cartesian basis to primitive cell basis

    The primitive cell basis uses lattice vectors as the coordinate system.
    This is the natural basis for describing crystal symmetries.

    Transformation formula for affine matrix [R|t]:
    - R_prim = (B^T)^(-1) @ R_cart @ B^T
    - t_prim = (B^T)^(-1) @ t_cart
    where B is the primitive lattice basis matrix (rows are lattice vectors)

    :param space_group_matrices_cartesian: Space group operators (affine) under Cartesian basis
    :param lattice_basis_primitive: Primitive cell basis (rows are lattice vectors in Cartesian coords)
    :return: Space group operators (affine) under primitive cell basis, shape (num_ops, 3, 4)
    """
    B = lattice_basis_primitive

    BT = B.T
    BT_inv = np.linalg.inv(BT)
    num_operators = len(space_group_matrices_cartesian)

    space_group_matrices_primitive = np.zeros((num_operators, 3, 4), dtype=float)
    for j in range(num_operators):
        # Transform rotation/reflection part
        space_group_matrices_primitive[j, :, 0:3] = BT_inv @ space_group_matrices_cartesian[j, :, 0:3] @ BT
        # Transform translation part
        space_group_matrices_primitive[j, :, 3] = BT_inv @ space_group_matrices_cartesian[j, :, 3]

    return space_group_matrices_primitive


# ==============================================================================
# STEP 6: Define orbital representation functions
# ==============================================================================
def space_group_representation_D_orbitals(R):
    """
    Compute how a symmetry operation acts on d orbitals

    Original function: GetSymD(R) in cd/SymGroup.py

    The d orbitals transform as quadratic functions of coordinates:
    d_xy, d_yz, d_zx, d_(x²-y²), d_(3z²-r²)

    This function computes the 5x5 representation matrix showing how
    the rotation R transforms the d orbital basis.

    :param R: Linear part of space group operation (3x3 rotation matrix) in Cartesian basis
    :return: Representation matrix (5x5) for d orbitals
    """
    [[R_11, R_12, R_13], [R_21, R_22, R_23], [R_31, R_32, R_33]] = R
    RD = np.zeros((5, 5))
    sr3 = np.sqrt(3)

    # Row 0: d_xy orbital transformation
    RD[0, 0] = R_11*R_22 + R_12*R_21
    RD[0, 1] = R_21*R_32 + R_22*R_31
    RD[0, 2] = R_11*R_32 + R_12*R_31
    RD[0, 3] = 2*R_11*R_12 + R_31*R_32
    RD[0, 4] = sr3*R_31*R_32

    # Row 1: d_yz orbital transformation
    RD[1, 0] = R_12*R_23 + R_13*R_22
    RD[1, 1] = R_22*R_33 + R_23*R_32
    RD[1, 2] = R_12*R_33 + R_13*R_32
    RD[1, 3] = 2*R_12*R_13 + R_32*R_33
    RD[1, 4] = sr3*R_32*R_33

    # Row 2: d_zx orbital transformation
    RD[2, 0] = R_11*R_23 + R_13*R_21
    RD[2, 1] = R_21*R_33 + R_23*R_31
    RD[2, 2] = R_11*R_33 + R_13*R_31
    RD[2, 3] = 2*R_11*R_13 + R_31*R_33
    RD[2, 4] = sr3*R_31*R_33

    # Row 3: d_(x²-y²) orbital transformation
    RD[3, 0] = R_11*R_21 - R_12*R_22
    RD[3, 1] = R_21*R_31 - R_22*R_32
    RD[3, 2] = R_11*R_31 - R_12*R_32
    RD[3, 3] = (R_11**2 - R_12**2) + 1/2*(R_31**2 - R_32**2)
    RD[3, 4] = sr3/2*(R_31**2 - R_32**2)

    # Row 4: d_(3z²-r²) orbital transformation
    RD[4, 0] = 1/sr3*(2*R_13*R_23 - R_11*R_21 - R_12*R_22)
    RD[4, 1] = 1/sr3*(2*R_23*R_33 - R_21*R_31 - R_22*R_32)
    RD[4, 2] = 1/sr3*(2*R_13*R_33 - R_11*R_31 - R_12*R_32)
    RD[4, 3] = 1/sr3*(2*R_13**2 - R_11**2 - R_12**2) + 1/sr3/2*(2*R_33**2 - R_31**2 - R_32**2)
    RD[4, 4] = 1/2*(2*R_33**2 - R_31**2 - R_32**2)

    return RD.T


def space_group_representation_F_orbitals(R):
    """
    Compute how a symmetry operation acts on f orbitals

    Original function: GetSymF(R) in cd/SymGroup.py

    The f orbitals transform as cubic functions of coordinates:
    fz³, fxz², fyz², fxyz, fz(x²-y²), fx(x²-3y²), fy(3x²-y²)

    This function computes the 7x7 representation matrix showing how
    the rotation R transforms the f orbital basis.

    :param R: Linear part of space group operation (3x3 rotation matrix) in Cartesian basis
    :return: Representation matrix (7x7) for f orbitals
    """
    sr3 = np.sqrt(3)
    sr5 = np.sqrt(5)
    sr15 = np.sqrt(15)

    # Define cubic monomials: x³, y³, z³, x²y, xy², x²z, xz², y²z, yz², xyz
    x1x2x3 = np.array([
        [1, 1, 1],  # x³
        [2, 2, 2],  # y³
        [3, 3, 3],  # z³
        [1, 1, 2],  # x²y
        [1, 2, 2],  # xy²
        [1, 1, 3],  # x²z
        [1, 3, 3],  # xz²
        [2, 2, 3],  # y²z
        [2, 3, 3],  # yz²
        [1, 2, 3]   # xyz
    ], int)

    # Compute how rotation R acts on cubic monomials
    # Rx1x2x3[i,j] = coefficient of monomial j in transformed monomial i
    Rx1x2x3 = np.zeros((10, 10))
    for i in range(10):
        n1, n2, n3 = x1x2x3[i]
        # Transform each cubic monomial by applying R to each factor
        Rx1x2x3[i, 0] = R[1-1, n1-1] * R[1-1, n2-1] * R[1-1, n3-1]  # x³
        Rx1x2x3[i, 1] = R[2-1, n1-1] * R[2-1, n2-1] * R[2-1, n3-1]  # y³
        Rx1x2x3[i, 2] = R[3-1, n1-1] * R[3-1, n2-1] * R[3-1, n3-1]  # z³
        # x²y (sum of all permutations)
        Rx1x2x3[i, 3] = (R[1-1, n1-1] * R[1-1, n2-1] * R[2-1, n3-1] +
                         R[1-1, n1-1] * R[2-1, n2-1] * R[1-1, n3-1] +
                         R[2-1, n1-1] * R[1-1, n2-1] * R[1-1, n3-1])
        # xy² (sum of all permutations)
        Rx1x2x3[i, 4] = (R[1-1, n1-1] * R[2-1, n2-1] * R[2-1, n3-1] +
                         R[2-1, n1-1] * R[2-1, n2-1] * R[1-1, n3-1] +
                         R[2-1, n1-1] * R[1-1, n2-1] * R[2-1, n3-1])
        # x²z (sum of all permutations)
        Rx1x2x3[i, 5] = (R[1-1, n1-1] * R[1-1, n2-1] * R[3-1, n3-1] +
                         R[1-1, n1-1] * R[3-1, n2-1] * R[1-1, n3-1] +
                         R[3-1, n1-1] * R[1-1, n2-1] * R[1-1, n3-1])
        # xz² (sum of all permutations)
        Rx1x2x3[i, 6] = (R[1-1, n1-1] * R[3-1, n2-1] * R[3-1, n3-1] +
                         R[3-1, n1-1] * R[3-1, n2-1] * R[1-1, n3-1] +
                         R[3-1, n1-1] * R[1-1, n2-1] * R[3-1, n3-1])
        # y²z (sum of all permutations)
        Rx1x2x3[i, 7] = (R[2-1, n1-1] * R[2-1, n2-1] * R[3-1, n3-1] +
                         R[2-1, n1-1] * R[3-1, n2-1] * R[2-1, n3-1] +
                         R[3-1, n1-1] * R[2-1, n2-1] * R[2-1, n3-1])
        # yz² (sum of all permutations)
        Rx1x2x3[i, 8] = (R[2-1, n1-1] * R[3-1, n2-1] * R[3-1, n3-1] +
                         R[3-1, n1-1] * R[3-1, n2-1] * R[2-1, n3-1] +
                         R[3-1, n1-1] * R[2-1, n2-1] * R[3-1, n3-1])
        # xyz (sum of all 6 permutations)
        Rx1x2x3[i, 9] = (R[1-1, n1-1] * R[2-1, n2-1] * R[3-1, n3-1] +
                         R[1-1, n1-1] * R[3-1, n2-1] * R[2-1, n3-1] +
                         R[2-1, n1-1] * R[1-1, n2-1] * R[3-1, n3-1] +
                         R[2-1, n1-1] * R[3-1, n2-1] * R[1-1, n3-1] +
                         R[3-1, n1-1] * R[1-1, n2-1] * R[2-1, n3-1] +
                         R[3-1, n1-1] * R[2-1, n2-1] * R[1-1, n3-1])

    # Matrix to express f orbitals as linear combinations of cubic monomials
    # Rows: fz³, fxz², fyz², fxyz, fz(x²-y²), fx(x²-3y²), fy(3x²-y²)
    # Columns: x³, y³, z³, x²y, xy², x²z, xz², y²z, yz², xyz
    F = np.array([
        [       0,        0,   1/sr15,        0,        0, -3/2/sr15,        0, -3/2/sr15,        0,        0],  # fz³
        [-1/2/sr5,        0,        0,        0, -1/2/sr5,        0,    2/sr5,        0,        0,        0],  # fxz²
        [       0, -1/2/sr5,        0, -1/2/sr5,        0,        0,        0,        0,    2/sr5,        0],  # fyz²
        [       0,        0,        0,        0,        0,        0,        0,        0,        0,        1],  # fxyz
        [       0,        0,        0,        0,        0,      1/2,        0,     -1/2,        0,        0],  # fz(x²-y²)
        [ 1/2/sr3,        0,        0,        0,   -sr3/2,        0,        0,        0,        0,        0],  # fx(x²-3y²)
        [       0, -1/2/sr3,        0,    sr3/2,        0,        0,        0,        0,        0,        0]   # fy(3x²-y²)
    ])

    # Transform f orbitals: FR = F @ Rx1x2x3
    FR = F @ Rx1x2x3  # Shape: (7, 10)

    # Matrix to convert back from cubic monomials to f orbitals
    # Rows: fz³, fxz², fyz², fxyz, fz(x²-y²), fx(x²-3y²), fy(3x²-y²)
    # Columns: x³, y³, z³, x²y, xy², x²z, xz², y²z, yz², xyz
    CF = np.array([
        [     0,      0,   sr15,      0,      0,      0,      0,      0,      0,      0],  # fz³
        [     0,      0,      0,      0,      0,      0,  sr5/2,      0,      0,      0],  # fxz²
        [     0,      0,      0,      0,      0,      0,      0,      0,  sr5/2,      0],  # fyz²
        [     0,      0,      0,      0,      0,      0,      0,      0,      0,      1],  # fxyz
        [     0,      0,      3,      0,      0,      2,      0,      0,      0,      0],  # fz(x²-y²)
        [ 2*sr3,      0,      0,      0,      0,      0,  sr3/2,      0,      0,      0],  # fx(x²-3y²)
        [     0, -2*sr3,      0,      0,      0,      0,      0,      0, -sr3/2,      0]   # fy(3x²-y²)
    ])

    # Final representation matrix for f orbitals
    RF = FR @ CF.T
    return RF.T


def space_group_representation_orbitals_all(space_group_matrices_cartesian):
    """
    Compute space group representations for all atomic orbital types

    Original function: GetSymOrb(SymXyz) in cd/SymGroup.py

    For each symmetry operation in the space group, compute how it transforms:
    - s orbitals (scalar, trivial representation)
    - p orbitals (3D vector: px, py, pz)
    - d orbitals (5D: dxy, dyz, dzx, d(x²-y²), d(3z²-r²))
    - f orbitals (7D: fz³, fxz², fyz², fxyz, fz(x²-y²), fx(x²-3y²), fy(3x²-y²))

    :param space_group_matrices_cartesian: Space group matrices (affine) under Cartesian basis
    :return: List of representations [repr_s, repr_p, repr_d, repr_f]
    """
    num_matrices, _, _ = space_group_matrices_cartesian.shape

    # S orbitals: spherically symmetric, trivial representation (all 1's)
    repr_s = np.ones((num_matrices, 1, 1))

    # P orbitals: transform as vectors (px, py, pz)
    # Use the rotation part of the space group matrices
    repr_p = copy.deepcopy(space_group_matrices_cartesian[:, :3, :3])

    # D orbitals: 5x5 representation
    # Basis: dxy, dyz, dzx, d(x²-y²), d(3z²-r²)
    repr_d = np.zeros((num_matrices, 5, 5))
    for i in range(num_matrices):
        R = space_group_matrices_cartesian[i, :3, :3]
        repr_d[i] = space_group_representation_D_orbitals(R)

    # F orbitals: 7x7 representation
    # Basis: fz³, fxz², fyz², fxyz, fz(x²-y²), fx(x²-3y²), fy(3x²-y²)
    repr_f = np.zeros((num_matrices, 7, 7))
    for i in range(num_matrices):
        R = space_group_matrices_cartesian[i, :3, :3]
        repr_f[i] = space_group_representation_F_orbitals(R)

    repr_s_p_d_f = [repr_s, repr_p, repr_d, repr_f]

    return repr_s_p_d_f


# ==============================================================================
# STEP 7: Read space group data and compute transformations
# ==============================================================================
# Path to database file containing all space group symmetry operations
in_space_group_file = "./read_only/space_group_matrices_Bilbao.txt"

# Read space group matrices from database (in Bilbao basis)
space_group_matrices = read_space_group(in_space_group_file, space_group)

# Transform to Cartesian basis
space_group_matrices_cartesian = space_group_to_cartesian_basis(space_group_matrices, space_group_basis)

# Transform to primitive cell basis
space_group_matrices_primitive = space_group_to_primitive_cell_basis(space_group_matrices_cartesian, lattice_basis_primitive)


# ==============================================================================
# STEP 8: Compute orbital representations
# ==============================================================================
# Compute how symmetry operations act on s, p, d, f orbitals
repr_s_p_d_f = space_group_representation_orbitals_all(space_group_matrices_cartesian)


# ==============================================================================
# STEP 9: Package results and output as JSON
# ==============================================================================
# Create output dictionary with all computed representations
space_group_representations = {
    # Bilbao space group matrices (original from database)
    "space_group_matrices": space_group_matrices.tolist(),

    # Space group matrices in Cartesian coordinates
    # Original variable: SymXyzt
    "space_group_matrices_cartesian": space_group_matrices_cartesian.tolist(),

    # Space group matrices in primitive cell basis
    # Original variable: SymLvSG
    "space_group_matrices_primitive": space_group_matrices_primitive.tolist(),

    # Orbital representations (s, p, d, f)
    # Original variable: SymOrb
    "repr_s_p_d_f": [
        repr_s_p_d_f[0].tolist(),  # s orbital representation
        repr_s_p_d_f[1].tolist(),  # p orbital representation
        repr_s_p_d_f[2].tolist(),  # d orbital representation
        repr_s_p_d_f[3].tolist()   # f orbital representation
    ]
    ,
    # Space group origin in different coordinate systems
    "space_group_origin_cartesian": space_group_origin_cart.tolist(),
    "space_group_origin_fractional_primitive": space_group_origin_frac_primitive.tolist()
}

# Output as JSON to stdout
print(json.dumps(space_group_representations, indent=2), file=sys.stdout)