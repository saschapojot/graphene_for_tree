from multiprocessing import Pool, cpu_count
import sympy as sp
import numpy as np
import sys
from pathlib import Path

from plot_energy_band.load_path_in_Brillouin_zone import  subroutine_get_interpolated_points_in_BZ_and_quantum_number_k
from load_Hk_parameters.load_Hk_and_hopping import subroutine_get_Hk

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


#this script loads symbolic Hk, loads path in Brillouin zone (BZ), diagonalize the Hk for each point in BZ path


def Hk_symbolic_to_np(Hk,processed_input_data):
    try:
        dim = processed_input_data["parsed_config"]['dim']
    except KeyError:
        raise KeyError("The processed_input_data dictionary is missing 'parsed_config' or 'dim'.")

    #check Hk's free symbols
    # Get the actual free symbols present in the symbolic matrix
    Hk_free_symbols = Hk.free_symbols
    # sp.pprint(hk_free_symbols)
    # the naming convention: k0, k1, k2
    k0 = sp.Symbol('k0', real=True)
    k1 = sp.Symbol('k1', real=True)
    k2 = sp.Symbol('k2', real=True)
    # Determine the expected variables based on dimensionality
    if dim == 1:
        expected_vars = [k0]
    elif dim == 2:
        expected_vars = [k0, k1]
    elif dim == 3:
        expected_vars = [k0, k1, k2]
    else:
        raise ValueError(f"Unsupported dimension: {dim}. Only 1, 2, or 3 are supported.")

    expected_vars_set = set(expected_vars)
    # Check for undefined symbols
    # We subtract the allowed k-points from the symbols found in Hk.
    # If anything remains (e.g., an unsubstituted 't' or 'mu', dimension not matched), it's an error.
    undefined_symbols = Hk_free_symbols - expected_vars_set
    if len(undefined_symbols) > 0:
        raise ValueError(
            f"The symbolic Hamiltonian contains undefined free symbols: {undefined_symbols}. "
            f"For dimension {dim}, only variables {expected_vars_set} are allowed. "
            "Please ensure all physical parameters (hopping amplitudes, onsite energies) "
            "have been substituted with numerical values."
        )

    # Convert to Numpy function (Lambdify)
    # The resulting function will accept arguments corresponding to expected_vars
    # e.g., if dim=2, func(k0_val, k1_val)
    Hk_np = sp.lambdify(expected_vars, Hk, modules='numpy')
    return Hk_np


def generate_Hk_matrix(Hk_np, quantum_numbers_k, processed_input_data):
    """
    Generates numerical Hamiltonian matrices for every k-point in the provided path.
    Args:
        Hk_np:  The lambdified (numpy-compatible) function of the Hamiltonian.
        quantum_numbers_k: A 2D numpy array where rows are k-points.
        processed_input_data: Dictionary containing configuration (like 'dim').

    Returns:
        Hk_matrices_all: A 3D numpy array of shape (N_k_points, Matrix_Dim, Matrix_Dim).

    """
    # 1. Determine the number of k-points (rows) to calculate
    n_row, n_col = quantum_numbers_k.shape
    #
    # print(f"n_row={n_row}, n_col={n_col}")
    # 2. Retrieve dimensionality (1D, 2D, or 3D) from the config
    try:
        dim = processed_input_data["parsed_config"]['dim']
    except KeyError:
        raise KeyError("The processed_input_data dictionary is missing 'parsed_config' or 'dim'.")

    # 3. Extract the relevant spatial coordinates.
    # the first 'dim' columns (e.g., k0, k1 for 2D).
    quantum_numbers_input = quantum_numbers_k[:, 0:dim]

    # Initialize a list to store the numerical Hamiltonian matrices
    Hk_matrices_list = []

    # 4. Iterate over each k-point (each row in the input)
    for i in range(n_row):
        # Get the specific k-point coordinates for this step
        k_point = quantum_numbers_input[i, :]

        # Pass the components of the k-point as separate arguments to the lambdified function.
        # The * operator unpacks the numpy array into positional arguments (k0, k1, etc.)
        H_k_numerical = Hk_np(*k_point)

        # Ensure the output is a numpy array (lambdify sometimes returns lists/scalars depending on backend)
        H_k_numerical = np.array(H_k_numerical, dtype=complex)

        Hk_matrices_list.append(H_k_numerical)

    # 6. Convert the list of matrices into a single 3D numpy array.
    # This stacks the matrices along a new first axis.
    # Final Shape: (n_k_points, matrix_dim, matrix_dim)
    # This format is required for efficient, vectorized diagonalization later.
    Hk_matrices_all = np.array(Hk_matrices_list)

    return Hk_matrices_all


# --- Parallel Diagonalization Functions ---
def diagonalize_chunk(matrix_chunk):
    """
     Worker function: Diagonalizes a subset (chunk) of matrices.
    Args:
        matrix_chunk:  matrix_chunk: A 3D numpy array of shape (n_chunk, dim, dim).

    Returns:
        eigenvalues_sorted: 2D array (n_chunk, matrix_dim), sorted ascending.
        eigenvectors_sorted: 3D array (n_chunk, matrix_dim, matrix_dim), columns sorted matching eigenvalues.
    """
    # 1. Diagonalize
    # np.linalg.eigh usually sorts by default, but we will enforce it below to be safe.
    eigenvalues_chunk, eigenvectors_chunk = np.linalg.eigh(matrix_chunk)
    # --- Explicit Sorting (Optional but safe) ---
    # 2. Get the indices that would sort the eigenvalues along the last axis (axis 1)
    # argsort returns indices of shape (n_chunk, dim)
    sort_indices = np.argsort(eigenvalues_chunk, axis=1)
    # 3. Reorder eigenvalues
    # We use take_along_axis to apply the sort indices to the 2D array
    eigenvalues_sorted = np.take_along_axis(eigenvalues_chunk, sort_indices, axis=1)
    # 4. Reorder eigenvectors
    # Eigenvectors are columns. The array shape is (n_chunk, row, col).
    # We need to sort the columns (axis 2) based on the eigenvalue indices.
    # We must expand dimensions of sort_indices to match the eigenvector shape: (n_chunk, 1, dim)
    sort_indices_expanded = sort_indices[:, np.newaxis, :]
    eigenvectors_sorted = np.take_along_axis(eigenvectors_chunk, sort_indices_expanded, axis=2)
    return eigenvalues_sorted, eigenvectors_sorted



def diagonalize_all_Hk_matrices(Hk_matrices_all, num_processes=None):
    """
    Parallelizes the diagonalization of the Hamiltonian matrices using multiprocessing.

    Args:
        Hk_matrices_all: 3D numpy array (n_k_points, dim, dim).
        num_processes: Number of CPU cores to use. If None, uses all available cores.

    Returns:
          all_eigenvalues: 2D array (n_k_points, dim)
          all_eigenvectors: 3D array (n_k_points, dim, dim)

    """
    # num_k_points = Hk_matrices_all.shape[0]
    # 1. Determine number of processes
    if num_processes is None:
        num_processes = cpu_count()
    print(f"Parallelism={num_processes}")
    # 2. Split the data into chunks
    # np.array_split divides the array into sub-arrays along axis 0.
    # It handles cases where n_k_points is not perfectly divisible by num_processes.
    chunks = np.array_split(Hk_matrices_all, num_processes, axis=0)
    # 3. Create the Pool and map the worker function
    with Pool(processes=num_processes) as pool:
        # pool.map applies 'diagonalize_chunk' to each item in 'chunks'
        # results is a list of tuples: [(evals_1, evecs_1), (evals_2, evecs_2), ...]
        results = pool.map(diagonalize_chunk, chunks)
    # 4. Reassemble the results
    # zip(*results) unzips the list of tuples into two separate tuples:
    # one containing all eigenvalue chunks, one containing all eigenvector chunks.
    eigenvalues_list, eigenvectors_list = zip(*results)
    # Concatenate the chunks back into monolithic numpy arrays
    all_eigenvalues = np.concatenate(eigenvalues_list, axis=0)
    all_eigenvectors = np.concatenate(eigenvectors_list, axis=0)
    return all_eigenvalues, all_eigenvectors




