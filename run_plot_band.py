from datetime import datetime
import sys

import numpy as np
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

Hk_np=Hk_symbolic_to_np(Hk,processed_input_data)

Hk_matrices_all=generate_Hk_matrix(Hk_np,quantum_numbers_k,processed_input_data)

one_chunk=Hk_matrices_all[:3]
h0=one_chunk[0]
h1=one_chunk[1]
h2=one_chunk[2]


vals0,vecs0=np.linalg.eigh(h0)

vals1,vecs1=np.linalg.eigh(h1)


vals2,vecs2=np.linalg.eigh(h2)

t_diag_start=datetime.now()
all_eigenvalues, all_eigenvectors=diagonalize_all_Hk_matrices(Hk_matrices_all,12)
t_diag_end=datetime.now()

print("time: ",t_diag_end-t_diag_start )
print(f"vals0={vals0}")
print(f"vals1={vals1}")
print(f"vals2={vals2}")

print(f"all_eigenvalues={all_eigenvalues}")