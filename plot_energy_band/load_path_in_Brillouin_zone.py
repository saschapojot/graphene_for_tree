import pickle
import re
import sys
from pathlib import Path
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