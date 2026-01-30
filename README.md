
#this is for demonstrationg during symmetry analysis development

###########################
Part I, proprocessing
1. python graphene_scripy.py ./path/to/xxx.conf
    (i) parse conf file
        ./parse_files/parse_conf.py
    (ii) check if conf file data are valid
        ./parse_files/sanity_check.py
    (iii)   read space group matrices (Bilbao),
            convert space group matrices (affine) from conventional basis to Cartesian basis
   (iv) ... (symmetry analysis)
this project is for testing symmetry analyzer core functionality

###########################
#general case
#step 1 gives Hamiltonian in k space, saved in pickle and tex
1. python general_script.py ./path/to/xxx.conf 

#step 2 deals with energy band plotting
# 2.1: set path endpoints' fractional coordinates in BZ
# 2.2: parse the path, load Hk, load hopping parameters
# 2.3: compute eigenvalues, save to file
# 2.4 load file from step 2.3, make plots
2. python 