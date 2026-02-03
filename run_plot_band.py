from datetime import datetime
import sys
import sympy as sp

sp.init_printing(use_unicode=False, wrap_line=False)
#self defined
from classes.class_defs import frac_to_cartesian, atomIndex, hopping, vertex, T_tilde_total
#loading Hk module

from load_Hk_parameters.load_Hk_and_hopping import *
from plot_energy_band.load_path_in_Brillouin_zone import *
from plot_energy_band.block_diagonalization import *

argErrCode = 20
if (len(sys.argv) != 2):
    print("wrong number of arguments")
    print("example: python preprocessing.py /path/to/mc.conf")
    exit(argErrCode)


confFileName = str(sys.argv[1])

Hk=subroutine_get_Hk(confFileName)


all_coords, all_distances, high_symmetry_indices, high_symmetry_labels,quantum_numbers_k,processed_input_data=subroutine_get_interpolated_points_in_BZ_and_quantum_number_k(confFileName)

Hk_symbolic_to_np(Hk,processed_input_data)