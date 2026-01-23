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


def validate_data_files(file_paths: dict, verbose: bool = False) -> list:
    """
    Check which data files exist.

    Args:
        file_paths: Dictionary mapping file types to file paths (from get_data_file_paths)
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

    for file_type, file_path_str in file_paths.items():
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
