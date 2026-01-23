from datetime import datetime
import sys

#self defined
from classes.class_defs import frac_to_cartesian, atomIndex, hopping, vertex, T_tilde_total
#loading Hk module

from load_Hk import load_hamiltonian_from_path


argErrCode = 20
if (len(sys.argv) != 2):
    print("wrong number of arguments")
    print("example: python preprocessing.py /path/to/mc.conf")
    exit(argErrCode)


confFileName = str(sys.argv[1])

hamiltonian_data=load_hamiltonian_from_path(confFileName,True)