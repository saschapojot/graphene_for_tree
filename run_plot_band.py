from datetime import datetime
import sys
import sympy as sp

sp.init_printing(use_unicode=False, wrap_line=False)
#self defined
from classes.class_defs import frac_to_cartesian, atomIndex, hopping, vertex, T_tilde_total
#loading Hk module

from load_Hk_parameters.load_Hk_and_hopping import *
from plot_energy_band.load_path_in_Brillouin_zone import *

argErrCode = 20
if (len(sys.argv) != 2):
    print("wrong number of arguments")
    print("example: python preprocessing.py /path/to/mc.conf")
    exit(argErrCode)


confFileName = str(sys.argv[1])

conf_dir=get_data_directory(confFileName)

file_paths=get_data_file_paths(conf_dir)

H_file=file_paths["hamiltonian_pickle"]
hop_file=file_paths["parameters"]

h,hop=load_hamiltonian_and_hopping_from_path(confFileName,True)
h_mat=h['hamiltonian']
# df=h_mat-h_mat.H
# sp.pprint(sp.simplify(df))
Hk=substitute_hopping_parameters(h,hop,True)


k_path_and_input_files=get_file_paths(conf_dir)
validate_k_path_file(k_path_and_input_files)
