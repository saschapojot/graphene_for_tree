
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
