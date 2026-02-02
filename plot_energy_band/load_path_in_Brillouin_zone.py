import pickle
import re
import sys
from pathlib import Path

import numpy as np

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


#this script loads  path in BZ

# ==============================================================================
# Path and Directory Utilities
# ==============================================================================
def get_data_directory(conf_file_path: str) -> str:
    """
    Extract the directory containing the configuration file and data files.
    Args:
        conf_file_path:  Path to the configuration file

    Returns:
         String path to the data directory
    Raises:
        FileNotFoundError: If configuration file doesn't exist

    """
    config_path = Path(conf_file_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {conf_file_path}")

    config_dir = config_path.parent
    return str(config_dir.resolve())


def get_file_paths(data_dir: str)-> dict:
    """

    Args:
        data_dir: data directory

    Returns: dict containing k_path file path and preprocessed_input_file path

    """
    data_path = Path(data_dir)
    return {
        "k_path_file": str(data_path / "k_path.conf"),
        "preprocessed_input_file": str(data_path/"preprocessed_input.pkl")
    }


def validate_k_path_file(file_paths_dict: dict) -> None:
    """
    Verify that the k-path configuration file and preprocessed input exist.
    Args:
        file_paths_dict: Dictionary containing paths to required files.

    Returns:
        None

     Raises:
        FileNotFoundError: If any of the files do not exist.
    """
    missing_files = []
    file_descriptions = {
        'k_path_file': "k-path endpoints",
        'preprocessed_input_file': "processed input parameters"
    }
    for key, description in file_descriptions.items():
        file_path = file_paths_dict.get(key)

        # Check if the path is None or if the file does not exist
        if not file_path or not Path(file_path).exists():
            missing_files.append(f"Missing {description} file at: {file_path}")

    if missing_files:
        error_message = "The following required files were not found:\n" + "\n".join(missing_files)
        raise FileNotFoundError(error_message)


# ==============================================================================
# Define regex patterns for parsing
# ==============================================================================
# General key=value pattern
key_value_pattern = r'^([^=\s]+)\s*=\s*([^=]*)\s*$'
# Pattern for floating point numbers (including scientific notation)
float_pattern = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"

#regex for fractional coordinates
# 1D: x (Single value)
fractional_coord_1d_pattern = rf"^\s*({float_pattern})\s*$"

# 2D: x, y (Comma separated)
fractional_coord_2d_pattern=rf"^\s*({float_pattern})\s*,\s*({float_pattern})\s*$"

# 3D: x, y, z (Comma separated, based on your input)
fractional_coord_3d_pattern = rf"^\s*({float_pattern})\s*,\s*({float_pattern})\s*,\s*({float_pattern})\s*$"

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

def parse_preprocessed_input(preprocessed_input_file_name):
    """
    Args:
        preprocessed_input_file_name: contains parsed information, is a pkl file

    Returns: a dictionary containing all parsed information
    """
    try:
        with open(preprocessed_input_file_name, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"The preprocessed input file was not found at: {preprocessed_input_file_name}")
    except pickle.UnpicklingError:
        raise ValueError(f"Error decoding the pickle file: {preprocessed_input_file_name}. It may be corrupted.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading {preprocessed_input_file_name}: {e}")


def read_k_path_conf(k_path_file_name: str, processed_input_data):
    """
    Reads the k-path configuration file and parses coordinates based on system dimensionality.

    Args:
        k_path_file_name: Path to the k_path.conf file.
        processed_input_data: Dictionary containing parsed input data (must contain 'dim').

    Returns:
        list: A list of dictionaries, where each dictionary represents a k-point:
              {'label': str, 'coords': [float, ...]}
    """
    # 1. Retrieve dimensionality
    try:
        dim = processed_input_data["parsed_config"]['dim']
    except KeyError:
        raise KeyError("The processed_input_data dictionary is missing 'parsed_config' or 'dim'.")

    # 2. Get cleaned lines from file
    linesWithCommentsRemoved = removeCommentsAndEmptyLines(k_path_file_name)

    parsed_k_points = []

    # 3. Parse each line
    for oneLine in linesWithCommentsRemoved:
        # Check if line matches key=value format
        matchLine = re.match(key_value_pattern, oneLine)

        if matchLine:
            key_label = matchLine.group(1).strip()
            value_str = matchLine.group(2).strip()



            # Validate and parse based on dimension
            if dim == 1:
                coord_match = re.match(fractional_coord_1d_pattern, value_str)
                if coord_match:
                    coords = [float(coord_match.group(1))]
                else:
                    raise ValueError(
                        f"Dimension mismatch: System is 1D, but value '{value_str}' for key '{key_label}' is not a valid 1D coordinate.")

            elif dim == 2:
                coord_match = re.match(fractional_coord_2d_pattern, value_str)
                if coord_match:
                    coords = [float(coord_match.group(1)), float(coord_match.group(2))]
                else:
                    raise ValueError(
                        f"Dimension mismatch: System is 2D, but value '{value_str}' for key '{key_label}' is not a valid 2D coordinate.")

            elif dim == 3:
                coord_match = re.match(fractional_coord_3d_pattern, value_str)
                if coord_match:
                    coords = [float(coord_match.group(1)), float(coord_match.group(2)), float(coord_match.group(3))]
                else:
                    raise ValueError(
                        f"Dimension mismatch: System is 3D, but value '{value_str}' for key '{key_label}' is not a valid 3D coordinate.")

            else:
                raise ValueError(
                    f"Unsupported dimension found in processed input: {dim}. Only 1, 2, or 3 are supported.")

            # If successful, append to results
            parsed_k_points.append({
                "label": key_label,
                "coords": coords
            })

        else:
            # Line doesn't match key=value format
            print("line: " + oneLine + " is discarded.", file=sys.stderr)

    return parsed_k_points


def compute_Brillouin_zone_basis(processed_input_data):
    parsed_config=processed_input_data["parsed_config"]
    basis = parsed_config.get('lattice_basis', [])
    a0,a1,a2=basis
    a0=np.array(a0)
    a1= np.array(a1)
    a2 = np.array(a2)

    #volume, may be signed if a0,a1,a2 do not have positive orientation
    Omega=np.dot(a0,np.cross(a1,a2))

    b0=2*np.pi*np.cross(a1,a2)/Omega

    b1=2*np.pi*np.cross(a2,a0)/Omega

    b2=2*np.pi*np.cross(a0,a1)/Omega

    return b0,b1,b2

def generate_interpolation(point_start_frac, point_end_frac,BZ_basis_vectors,interpolate_point_num=15):


    # 1. Convert Fractional to Cartesian
    # We use zip to pair the coordinate component (u, v, w) with the basis vector (b0, b1, b2)
    # This automatically handles 1D, 2D, or 3D depending on the length of the inputs.
    start_cart = sum(c * b for c, b in zip(point_start_frac, BZ_basis_vectors))
    end_cart = sum(c * b for c, b in zip(point_end_frac, BZ_basis_vectors))
    # 2. Linear Interpolation
    # Create a parameter t going from 0 to 1
    t = np.linspace(0, 1, interpolate_point_num)
    # Vector from start to end
    vector_diff = end_cart - start_cart
    # Calculate path: Start + t * (End - Start)
    # np.outer allows us to multiply the shape (N,) t array by the shape (3,) vector
    #each row is an interpolated point
    interpolated_cart_coords = start_cart + np.outer(t, vector_diff)
    # 3. Calculate Distances
    # Euclidean distance of the full segment
    segment_length = np.linalg.norm(vector_diff)
    distances = t * segment_length

    return interpolated_cart_coords, distances


def interpolate_path(parsed_k_points, processed_input_data, interpolate_point_num=15):
    """
    Interpolates between consecutive k-points to create a continuous path in reciprocal space.

    Args:
        parsed_k_points: List of dicts, each containing 'label' and 'coords' for high-symmetry points.
        processed_input_data: Dictionary containing system configuration (lattice basis, dim).
        interpolate_point_num: Number of points to generate per segment.

    Returns:
        tuple: (all_coords, all_distances, high_symmetry_indices, high_symmetry_labels)
               - all_coords: (N, 3) array of Cartesian coordinates along the path.
               - all_distances: (N,) array of cumulative distances along the path.
               - high_symmetry_indices: List of indices in all_coords corresponding to the input k-points.
               - high_symmetry_labels: List of labels corresponding to high_symmetry_indices.
    """
    # 1. Get Reciprocal Lattice Basis Vectors
    b0, b1, b2 = compute_Brillouin_zone_basis(processed_input_data)
    dim = processed_input_data["parsed_config"]['dim']

    if dim == 1:
        BZ_basis_vectors = [b0]
    elif dim == 2:
        BZ_basis_vectors = [b0, b1]
    elif dim == 3:
        BZ_basis_vectors = [b0, b1, b2]
    else:
        raise ValueError(f"Unsupported dimension: {dim}. Only 1, 2, or 3 are supported.")

    if len(parsed_k_points) < 2:
        raise ValueError("At least two k-points are required to interpolate a path.")

    all_coords = []
    all_distances = []
    high_symmetry_indices = []
    high_symmetry_labels = []

    cumulative_distance = 0.0
    current_index_count = 0

    # 2. Loop through consecutive pairs
    for i in range(len(parsed_k_points) - 1):
        start_point = parsed_k_points[i]
        end_point = parsed_k_points[i + 1]

        start_frac = start_point['coords']
        end_frac = end_point['coords']

        # Call the helper function
        # Note: generate_interpolation returns (coords, distances_from_start_of_segment)
        segment_coords, segment_distances = generate_interpolation(
            start_frac,
            end_frac,
            BZ_basis_vectors,
            interpolate_point_num
        )

        # 3. Accumulate Data
        # For the very first point of the entire path, we add everything.
        # For subsequent segments, we skip the first point to avoid duplication
        # because the end of segment i is the start of segment i+1.
        if i == 0:
            # Record the start label
            high_symmetry_indices.append(current_index_count)
            high_symmetry_labels.append(start_point['label'])

            # Add all points
            all_coords.append(segment_coords)
            # Add cumulative distance offset
            all_distances.append(segment_distances + cumulative_distance)

            current_index_count += len(segment_coords)
        else:
            # Skip the first point (it overlaps with previous segment's last point)
            all_coords.append(segment_coords[1:])

            # Add cumulative distance offset, skipping first distance
            all_distances.append(segment_distances[1:] + cumulative_distance)

            current_index_count += len(segment_coords) - 1

        # Update cumulative distance for the next segment
        # segment_distances[-1] is the length of the current segment
        cumulative_distance += segment_distances[-1]

        # Record the end label of this segment
        high_symmetry_indices.append(current_index_count - 1)
        high_symmetry_labels.append(end_point['label'])

    # 4. Concatenate arrays
    all_coords = np.vstack(all_coords)
    all_distances = np.concatenate(all_distances)

    return all_coords, all_distances, high_symmetry_indices, high_symmetry_labels








