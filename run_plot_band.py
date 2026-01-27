from datetime import datetime
import sys
import sympy as sp

#self defined
from classes.class_defs import frac_to_cartesian, atomIndex, hopping, vertex, T_tilde_total
#loading Hk module

from load_Hk_parameters.load_Hk_and_hopping import *


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
substitute_hopping_parameters(h,hop,True)
