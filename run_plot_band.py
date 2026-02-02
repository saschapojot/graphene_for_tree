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
k_path_file_name=k_path_and_input_files["k_path_file"]
processed_input_file_name=k_path_and_input_files["preprocessed_input_file"]

processed_input_data=parse_preprocessed_input(processed_input_file_name)
parsed_k_points=read_k_path_conf(k_path_file_name,processed_input_data)
b0,b1,b2=compute_Brillouin_zone_basis(processed_input_data)
