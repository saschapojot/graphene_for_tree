"""
- Utility module for loading and working with tight-binding Hamiltonians
This module provides functions to:
- Load saved Hamiltonian data from pickle files
- Load hopping parameters from text files
- Substitute parameters into symbolic Hamiltonians

"""
from pathlib import Path
import pickle
import numpy as np
import sympy as sp
import re

# Exit codes for different error types
fmtErrStr = "format error: "
formatErrCode = 1        # Format/syntax errors in conf file
valueMissingCode = 2     # Required values are missing
paramErrCode = 3         # Wrong command-line parameters
fileNotExistErrCode = 4  # Configuration file doesn't exist

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



def get_data_file_paths(data_dir: str) -> dict:
    """
     Construct paths to all expected data files in the directory.

    Args:
        data_dir:  Directory containing the data files (as string)

    Returns:
        Dictionary mapping file types to file paths (as strings)

    """
    data_path = Path(data_dir)
    return {
        'hamiltonian_pickle': str(data_path / "hamiltonian_data.pkl"),
        'parameters': str(data_path / "hopping_parameters.txt"),
    }


def validate_data_files(file_paths_dict: dict, verbose: bool = False) -> list:
    """
    Check which data files exist.

    Args:
        file_paths_dict: Dictionary mapping file types to file paths (from get_data_file_paths)
        verbose: If True, print status messages

    Returns:
        List of missing file types
    """
    missing_files = []

    if verbose:
        print("=" * 80)
        print("CHECKING FOR DATA FILES")
        print("=" * 80)

    file_descriptions = {
        'hamiltonian_pickle': "Hamiltonian data (pickle)",
        'parameters': "Hopping parameters"
    }

    for file_type, file_path_str in file_paths_dict.items():
        file_path = Path(file_path_str)
        exists = file_path.exists()

        if verbose:
            status = "✓" if exists else "✗"
            desc = file_descriptions.get(file_type, file_type)
            status_text = "" if exists else " [MISSING]"
            print(f"{status} {desc}: {file_path.name}{status_text}")

        if not exists:
            missing_files.append(file_type)

    if verbose:
        print("=" * 80)

    return missing_files

# ==============================================================================
# Define regex patterns for parsing
# ==============================================================================
# General key=value pattern
key_value_pattern = r'^([^=\s]+)\s*=\s*([^=]*)\s*$'
# Pattern for floating point numbers (including scientific notation)
float_pattern = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
# Pattern for pure real numbers (float)
# Examples: 2.3, -3.5, 1e-3, -4.2e-3, 5, -10
pure_float_pattern = rf"({float_pattern})"
# Pattern for pure imaginary numbers: imag*j or imag*i
# Examples: 2.3j, -3e-5j, 2i, -4.2e-3i
pure_imaginary_pattern = rf"({float_pattern})[ij]"
# Pattern for complex numbers: real + imag*j or real + imag*i
# Examples: 1.5+2.3j, -1.0-3e-5j, 1+2i, -3.5e2+4.2e-3i
complex_pattern = rf"({float_pattern})\s*([\+\-])\s*({float_pattern})[ij]"
# Pattern for orbital names (includes principal quantum number)
# Examples: 1s, 2s, 2px, 2py, 2pz, 3dxy, 3dxz, 3dyz, 3dx2-y2, 3dz2, etc.
orbital_pattern = r"[1-7](?:s|px|py|pz|dxy|dxz|dyz|dx2-y2|dz2|fxyz|fxz2|fyz2|fx3-3xy2|f3yx2-y3|fzx2-zy2|fz3)"

# Pattern for Independent complex Hopping Parameters (T)
# Examples: T^{5}_{2py,2s}, T^{2}_{2s,2py}, T^{0}_{3dxy,2px}
T_pattern = rf"T\^{{(\d+)}}_{{({orbital_pattern}),({orbital_pattern})}}"


# Pattern for Independent real Parts of Hopping Parameters (re_T)
# Examples: re_T^{0}_{2pz,2pz}, re_T^{1}_{2s,2py}, re_T^{2}_{2py,2px}
re_T_pattern = rf"re_T\^{{(\d+)}}_{{({orbital_pattern}),({orbital_pattern})}}"

# Pattern for Independent imaginary Parts of Hopping Parameters (im_T)
# Examples: im_T^{1}_{2s,2py}, im_T^{2}_{2s,2px}, im_T^{4}_{2py,2px}
im_T_pattern = rf"im_T\^{{(\d+)}}_{{({orbital_pattern}),({orbital_pattern})}}"
# ==============================================================================
# Define helper function to clean file contents
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
# Helper function to parse numeric values
# ==============================================================================
def parse_numeric_value(value_str):
    """
    Parse a numeric value that could be complex, pure imaginary, or pure real.
    Args:
        value_str:  String representation of the number

    Returns:
        tuple: (parsed_value, value_type) where value_type is one of:
            - 'complex': for numbers with both real and imaginary parts
            - 'pure_imaginary': for pure imaginary numbers (0 + bi)
            - 'float': for pure real numbers
            - (None, None): if parsing fails
     Examples:
          "2.3" -> (2.3, 'float')
        "2.3j" -> (2.3j, 'pure_imaginary')
        "1.5+2.3j" -> (1.5+2.3j, 'complex')
        "1.5-2.3i" -> (1.5-2.3j, 'complex')
        "-3e-5j" -> (-3e-5j, 'pure_imaginary')
    """
    value_str = value_str.strip()
    # Try to parse as complex number (real + imaginary)
    # Examples: 1.5+2.3j, -1.0-3e-5j, 1+2i
    match = re.match(complex_pattern, value_str)
    if match:
        real_part = float(match.group(1))
        sign = match.group(2)
        imag_abs = float(match.group(3))
        if sign == '+':
            return (complex(real_part, imag_abs), 'complex')
        else:  # sign == '-'
            return (complex(real_part, -imag_abs), 'complex')

    # Try to parse as pure imaginary number
    # Examples: 2.3j, -3e-5j, 2i, -4.2e-3i
    match = re.match(pure_imaginary_pattern, value_str)
    if match:
        imag_part = float(match.group(1))
        return (complex(0, imag_part), 'pure_imaginary')

    # Try to parse as pure real number
    # Examples: 2.3, -3.5, 1e-3, -4.2e-3
    match = re.match(pure_float_pattern, value_str)
    if match:
        return (float(match.group(1)), 'float')
    return (None, None)



def parse_hopping_parameters(hopping_parameters_file_name):
    """
    Parse hopping parameters from a text file and create SymPy symbols.
    The file should contain lines in the format:
    - T^{n}_{orbital1,orbital2} = complex_value or pure float or pure imaginary
    - re_T^{n}_{orbital1,orbital2} = float_value (real part)
    - im_T^{n}_{orbital1,orbital2} = float_value i or float_value j

    Args:
        hopping_parameters_file_name: Path to the hopping parameters file

    Returns:
        Dictionary containing:
        {
            'symbols': {
                'T': {(n, orbital1, orbital2): sympy.Symbol, ...},
                're_T': {(n, orbital1, orbital2): sympy.Symbol, ...},
                'im_T': {(n, orbital1, orbital2): sympy.Symbol, ...}
            },
            'values': {
                'T': {(n, orbital1, orbital2): complex_value, ...},
                're_T': {(n, orbital1, orbital2): float_value, ...},
                'im_T': {(n, orbital1, orbital2): pure_imaginary_value, ...}
            },
            'value_types': {
                'T': {(n, orbital1, orbital2): 'complex'|'pure_imaginary'|'float', ...},
                're_T': {(n, orbital1, orbital2): 'float', ...},
                'im_T': {(n, orbital1, orbital2): 'pure_imaginary', ...}
            },
            'substitution_dict': {sympy.Symbol: numeric_value, ...}
        }

    """
    # Check if file exists
    param_file = Path(hopping_parameters_file_name)
    if not param_file.exists():
        raise FileNotFoundError(f"Hopping parameters file not found: {hopping_parameters_file_name}")
    # Get cleaned lines from file
    linesWithCommentsRemoved = removeCommentsAndEmptyLines(hopping_parameters_file_name)

    # Initialize storage dictionaries
    symbols = {
        'T': {},  # SymPy symbols for complex hopping parameters
        're_T': {},  # SymPy symbols for real parts
        'im_T': {}  # SymPy symbols for imaginary parts
    }
    values = {
        'T': {},  # Numeric values for complex hopping parameters
        're_T': {},  # Numeric values for real parts (float)
        'im_T': {}  # Numeric values for imaginary parts (pure imaginary)
    }
    value_types = {
        'T': {},  # Types of values for complex hopping parameters
        're_T': {},  # Types for real parts (should always be 'float')
        'im_T': {}  # Types for imaginary parts (should always be 'pure_imaginary')
    }
    # This dictionary maps symbols to values for easy substitution into Hamiltonian
    substitution_dict = {}
    # Process each line
    for line_num, line in enumerate(linesWithCommentsRemoved, start=1):
        # Try to match key=value pattern
        kv_match = re.match(key_value_pattern, line)
        if not kv_match:
            print(f"Warning: Line {line_num} doesn't match key=value format: {line}")
            continue
        key = kv_match.group(1).strip()
        value_str = kv_match.group(2).strip()

        # ==============================================================================
        # Try to parse T^{n}_{orbital1,orbital2} = complex_value or pure float or pure imaginary
        # ==============================================================================
        T_match = re.match(T_pattern, key)
        if T_match:
            tree_idx = T_match.group(1)
            orbital1 = T_match.group(2)
            orbital2 = T_match.group(3)
            # Check if value is empty or whitespace only
            if not value_str:
                print(
                    f"Warning: Line {line_num} has empty value for T^{{{tree_idx}}}_{{{orbital1},{orbital2}}}, setting to 0")
                # Create the key tuple
                param_key = (tree_idx, orbital1, orbital2)
                # Create SymPy symbol: T^{n}_{orbital1,orbital2}
                symbol = sp.symbols(f'T^{{{tree_idx}}}_{{{orbital1},{orbital2}}}')
                # Store symbol with value 0
                symbols['T'][param_key] = symbol
                values['T'][param_key] = 0
                value_types['T'][param_key] = 'float'
                substitution_dict[symbol] = 0
                continue
            # Parse the value
            value, val_type = parse_numeric_value(value_str)
            if value is not None:
                # Create the key tuple
                param_key = (tree_idx, orbital1, orbital2)
                # Create SymPy symbol: T^{n}_{orbital1,orbital2}
                symbol = sp.symbols(f'T^{{{tree_idx}}}_{{{orbital1},{orbital2}}}')
                # Store symbol, value, and type
                symbols['T'][param_key] = symbol
                values['T'][param_key] = value
                value_types['T'][param_key] = val_type
                substitution_dict[symbol] = value
            else:
                raise ValueError(f"Line {line_num} has invalid value for T: {value_str}")
            continue

        # ==============================================================================
        # Try to parse re_T^{n}_{orbital1,orbital2} = float_value
        # ==============================================================================
        re_T_match = re.match(re_T_pattern, key)
        if re_T_match:
            tree_idx = re_T_match.group(1)
            orbital1 = re_T_match.group(2)
            orbital2 = re_T_match.group(3)
            # Check if value is empty or whitespace only
            if not value_str:
                print(
                    f"Warning: Line {line_num} has empty value for re_T^{{{tree_idx}}}_{{{orbital1},{orbital2}}}, setting to 0")
                # Create the key tuple
                param_key = (tree_idx, orbital1, orbital2)
                # Create SymPy symbol: re_T^{n}_{orbital1,orbital2}
                symbol = sp.symbols(f're_T^{{{tree_idx}}}_{{{orbital1},{orbital2}}}', real=True)
                # Store symbol with value 0
                symbols['re_T'][param_key] = symbol
                values['re_T'][param_key] = 0
                value_types['re_T'][param_key] = 'float'
                substitution_dict[symbol] = 0
                continue
            # Parse the value (should be a real number)
            value, val_type = parse_numeric_value(value_str)
            if value is not None:
                # Validate that it's a float type
                if val_type != 'float':
                    raise ValueError(
                        f"Line {line_num}: re_T^{{{tree_idx}}}_{{{orbital1},{orbital2}}} should have a real (float) value, but got {val_type}: {value_str}")
                # Create the key tuple
                param_key = (tree_idx, orbital1, orbital2)
                # Create SymPy symbol: re_T^{n}_{orbital1,orbital2}
                symbol = sp.symbols(f're_T^{{{tree_idx}}}_{{{orbital1},{orbital2}}}', real=True)
                # Store symbol, value, and type
                symbols['re_T'][param_key] = symbol
                values['re_T'][param_key] = float(value)
                value_types['re_T'][param_key] = 'float'
                substitution_dict[symbol] = float(value)
            else:
                raise ValueError(f"Line {line_num} has invalid value for re_T: {value_str}")
            continue

        # ==============================================================================
        # Try to parse im_T^{n}_{orbital1,orbital2} = float_value i or float_value j
        # ==============================================================================
        im_T_match = re.match(im_T_pattern, key)
        if im_T_match:
            tree_idx = im_T_match.group(1)
            orbital1 = im_T_match.group(2)
            orbital2 = im_T_match.group(3)
            # Check if value is empty or whitespace only
            if not value_str:
                print(
                    f"Warning: Line {line_num} has empty value for im_T^{{{tree_idx}}}_{{{orbital1},{orbital2}}}, setting to 0")
                # Create the key tuple
                param_key = (tree_idx, orbital1, orbital2)
                # Create SymPy symbol: im_T^{n}_{orbital1,orbital2}
                symbol = sp.symbols(f'im_T^{{{tree_idx}}}_{{{orbital1},{orbital2}}}')
                # Store symbol with value 0
                symbols['im_T'][param_key] = symbol
                values['im_T'][param_key] = 0
                value_types['im_T'][param_key] = 'pure_imaginary'
                substitution_dict[symbol] = 0
                continue
            # Parse the value (should be pure imaginary with i or j)
            value, val_type = parse_numeric_value(value_str)
            if value is not None:
                # Validate that it's a pure imaginary type or zero (float)
                # Accept pure_imaginary or float (if float, it should be 0)
                if val_type == 'pure_imaginary':
                    # Standard pure imaginary case
                    param_key = (tree_idx, orbital1, orbital2)
                    symbol = sp.symbols(f'im_T^{{{tree_idx}}}_{{{orbital1},{orbital2}}}')
                    # Store symbol, value, and type (value is already pure imaginary)
                    symbols['im_T'][param_key] = symbol
                    values['im_T'][param_key] = value
                    value_types['im_T'][param_key] = 'pure_imaginary'
                    substitution_dict[symbol] = value
                elif val_type == 'float' and value == 0:
                    # Accept zero as a valid imaginary part
                    param_key = (tree_idx, orbital1, orbital2)
                    symbol = sp.symbols(f'im_T^{{{tree_idx}}}_{{{orbital1},{orbital2}}}')
                    symbols['im_T'][param_key] = symbol
                    values['im_T'][param_key] = 0
                    value_types['im_T'][param_key] = 'pure_imaginary'
                    substitution_dict[symbol] = 0
                else:
                    raise ValueError(
                        f"Line {line_num}: im_T^{{{tree_idx}}}_{{{orbital1},{orbital2}}} should have a pure imaginary value or 0, but got {val_type}: {value_str}")
            else:
                raise ValueError(f"Line {line_num} has invalid value for im_T: {value_str}")
            continue

        # If we reach here, the key didn't match any expected pattern
        raise ValueError(f"Line {line_num} has unrecognized parameter format: {key}")

    return {
        'symbols': symbols,
        'values': values,
        'value_types': value_types,
        'substitution_dict': substitution_dict
    }
# ==============================================================================
# Hamiltonian Data Loading
# ==============================================================================
def load_hamiltonian_data(pickle_path: str, verbose: bool = True) -> dict:
    """
    Load Hamiltonian data from pickle file.
    Args:
        pickle_path: Path to hamiltonian_data.pkl file
        verbose: If True, print loading information

    Returns:
         Dictionary containing all Hamiltonian data:
        {
            'hamiltonian': sympy.Matrix - The symbolic Hamiltonian
            'hamiltonian_dimension': int - Total dimension
            'unit_cell_atoms': list - atomIndex objects
            'sorted_wyckoff_instance_ids': list - Ordered atom IDs
            'block_dimensions': dict - Block sizes {atom_id: num_orbitals}
            'T_tilde_blocks': dict - Individual hopping blocks
            'lattice_basis': np.ndarray - Lattice vectors
            'space_group_origin_cartesian': np.ndarray - Origin
            'config': dict - Configuration data
            'k_symbols': list - ['k0', 'k1', 'k2']
            'creation_date': str - ISO format timestamp
            'version': str - Data format version
        }
    Raises:
        FileNotFoundError: If pickle file doesn't exist
        pickle.UnpicklingError: If file is corrupted
    """
    pickle_file = Path(pickle_path)
    if not pickle_file.exists():
        raise FileNotFoundError(f"Hamiltonian data file not found: {pickle_path}")

    if verbose:
        print("\n" + "=" * 80)
        print("LOADING HAMILTONIAN DATA")
        print("=" * 80)
        print(f"Loading from: {pickle_file.name}")

    try:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)

        if verbose:
            print(f"✓ Successfully loaded Hamiltonian data")
            print(f"\nData Summary:")
            print(f"  Version: {data.get('version', 'unknown')}")
            print(f"  Creation date: {data.get('creation_date', 'unknown')}")
            print(f"  Hamiltonian dimension: {data['hamiltonian_dimension']}")
            print(f"  Number of unit cell atoms: {len(data['unit_cell_atoms'])}")
            print(f"  Number of hopping blocks: {len(data['T_tilde_blocks'])}")


        return data

    except pickle.UnpicklingError as e:
        raise pickle.UnpicklingError(f"Failed to load pickle file. File may be corrupted: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error loading Hamiltonian data: {e}")


def load_hamiltonian_and_hopping_from_path(conf_file_path: str, verbose: bool = True) -> dict:
    """
    Complete pipeline to load Hamiltonian data and hopping parameters from a configuration file path.



    """
    # ==============================================================================
    # STEP 1: Validate input path
    # ==============================================================================
    conf_path = Path(conf_file_path)
    if not conf_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {conf_file_path}")
    if verbose:
        print("\n" + "=" * 80)
        print("LOADING HAMILTONIAN DATA")
        print("=" * 80)
        print(f"Configuration file: {conf_path}")

    # ==============================================================================
    # STEP 2: Determine data directory and construct file paths
    # ==============================================================================
    # The Hamiltonian data is stored in the same directory as the config file
    try:
        data_dir = get_data_directory(conf_file_path)
        if verbose:
            print(f"Data directory: {data_dir}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error locating data directory: {e}")

    # Get paths to all expected data files
    file_paths = get_data_file_paths(data_dir)
    if verbose:
        print(f"\nExpected data files:")
        print(f"  Hamiltonian pickle: {Path(file_paths['hamiltonian_pickle']).name}")
        print(f"  Parameters file: {Path(file_paths['parameters']).name}")

    # ==============================================================================
    # STEP 3: Validate that required files exist
    # ==============================================================================
    missing_files = validate_data_files(file_paths, verbose=verbose)
    # Check if the essential Hamiltonian pickle file is missing
    if 'hamiltonian_pickle' in missing_files:
        raise FileNotFoundError(
            f"Required Hamiltonian data file not found: {file_paths['hamiltonian_pickle']}\n"
            f"Please run the main calculation script first to generate the data."
        )
    # Check if the parameters file is missing
    if 'parameters' in missing_files:
        raise FileNotFoundError(
            f"Required parameters file not found: {file_paths['parameters']}\n"
            f"Please run the main calculation script first to generate the parameters file."
        )

    # ==============================================================================
    # STEP 4: Load Hamiltonian data from pickle file
    # ==============================================================================
    try:
        hamiltonian_data = load_hamiltonian_data(
            file_paths['hamiltonian_pickle'],
            verbose=verbose
        )
    except Exception as e:
        raise Exception(f"Failed to load Hamiltonian data: {e}")

    # ==============================================================================
    # STEP 5: Extract and validate dimensionality information
    # ==============================================================================
    # Determine system dimensionality from k_symbols
    k_symbols = hamiltonian_data.get('k_symbols', [])
    if not k_symbols:
        raise ValueError(
            f"Missing 'k_symbols' in Hamiltonian data.\n"
            f"The data file may be corrupted or from an incompatible version.\n"
            f"Please regenerate the Hamiltonian data by running the main calculation script."
        )
    # Validate dimensionality consistency
    dim = len(k_symbols)
    config_dim = hamiltonian_data.get('config', {}).get('dim', None)
    if config_dim is not None and config_dim != dim:
        raise ValueError(
            f"Dimensionality mismatch: k_symbols indicates {dim}D "
            f"but config specifies {config_dim}D"
        )
    if verbose:
        print(f"\nSystem dimensionality: {dim}D")
        print(f"k-vector components: {', '.join(k_symbols)}")
    # ==============================================================================
    # STEP 6: Load and parse hopping parameters
    # ==============================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("LOADING HOPPING PARAMETERS")
        print("=" * 80)
        print(f"Loading from: {Path(file_paths['parameters']).name}")

    try:
        hopping_params = parse_hopping_parameters(file_paths['parameters'])
        if verbose:
            print(f"✓ Successfully loaded hopping parameters")
            print(f"\nParameter Summary:")
            print(f"  T parameters: {len(hopping_params['symbols']['T'])}")
            print(f"  re_T parameters: {len(hopping_params['symbols']['re_T'])}")
            print(f"  im_T parameters: {len(hopping_params['symbols']['im_T'])}")
            print(f"  Total symbols: {len(hopping_params['substitution_dict'])}")

    except Exception as e:
        raise Exception(f"Failed to load hopping parameters: {e}")

    # Add hopping parameters to the returned dictionary
    hamiltonian_data['hopping_parameters'] = hopping_params
    # ==============================================================================
    # STEP 7: Final validation and summary
    # ==============================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("LOADING COMPLETE")
        print("=" * 80)
        print(f"✓ Hamiltonian dimension: {hamiltonian_data['hamiltonian_dimension']}")
        print(f"✓ Number of atoms: {len(hamiltonian_data['unit_cell_atoms'])}")
        print(f"✓ System dimensionality: {dim}D")
        print(f"✓ Data version: {hamiltonian_data.get('version', 'unknown')}")
        print(f"✓ Hopping parameters loaded: {len(hopping_params['substitution_dict'])} total")
        print("=" * 80 + "\n")

    return hamiltonian_data






