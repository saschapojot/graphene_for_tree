"""
load_Hk.py - Utility module for loading and working with tight-binding Hamiltonians
This module provides functions to:
- Load saved Hamiltonian data from pickle files
- Load hopping parameters from text files
- Substitute parameters into symbolic Hamiltonians

"""
from pathlib import Path
import pickle
import numpy as np
import sympy as sp


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


def load_hamiltonian_from_path(conf_file_path: str, verbose: bool = True) -> dict:
    """
    Complete pipeline to load Hamiltonian data from a configuration file path.

    This function orchestrates the entire loading process:
     1. Extracts the data directory from the config file path
     2. Constructs paths to all expected data files
     3. Validates that required files exist
     4. Loads the Hamiltonian data from the pickle file
    Args:
        conf_file_path: Path to the configuration file (e.g., "path/to/hBN.conf")
        verbose: If True, print status messages during loading (default: True)

    Returns:
         dict: Complete Hamiltonian data containing:
            - 'hamiltonian': sympy.Matrix - The symbolic Hamiltonian H(k)
            - 'hamiltonian_dimension': int - Total dimension of H
            - 'unit_cell_atoms': list - atomIndex objects
            - 'sorted_wyckoff_instance_ids': list - Ordered atom IDs
            - 'block_dimensions': dict - Block sizes {atom_id: num_orbitals}
            - 'T_tilde_blocks': dict - Individual hopping blocks
            - 'lattice_basis': np.ndarray - Primitive lattice vectors
            - 'config': dict - Original configuration data
            - 'k_symbols': list of sympy.Symbol - k-vector components
                * 1D: [k0]
                * 2D: [k0, k1]
                * 3D: [k0, k1, k2]
            - 'creation_date': str - ISO format timestamp
            - 'version': str - Data format version

    Raises:
        FileNotFoundError: If config file or Hamiltonian pickle file doesn't exist
        ValueError: If dimensionality is invalid or data is corrupted

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
    # STEP 6: Final validation and summary
    # ==============================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("LOADING COMPLETE")
        print("=" * 80)
        print(f"✓ Hamiltonian dimension: {hamiltonian_data['hamiltonian_dimension']}")
        print(f"✓ Number of atoms: {len(hamiltonian_data['unit_cell_atoms'])}")
        print(f"✓ System dimensionality: {dim}D")
        print(f"✓ Data version: {hamiltonian_data.get('version', 'unknown')}")
        print("=" * 80 + "\n")

    return hamiltonian_data