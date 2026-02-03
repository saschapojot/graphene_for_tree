from multiprocessing import Pool
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