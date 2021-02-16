import sys
sys.path.append('../astroSABER/')

import os
import numpy as np
from astropy.io import fits

from hisa import HisaExtraction


###HI data to extract HISA
image_HI = 'HI_THOR_test_cube.fits'


###initialize hisa extraction
hisa = HisaExtraction(fitsfile=image_HI)

###path to noise map (or universal noise value)
#hisa.path_to_noise_map = os.path.join('.', 'dir', 'sub', '*.fits')
hisa.noise = 4. #Kelvin


###asymmetric least squares smoothing (Eilers et al. 2005) parameters
hisa.lam1 = 0.50
hisa.p1 = 0.90
hisa.lam2 = 0.50
hisa.p2 = 0.90


###maximum number of iterations (this limit is usually reached for strong continuum sources)
hisa.niters = 20


###this runs the hisa extraction routine
hisa.saber()



'''
the output will be three files:
hisa background spectrum (.fits)
hisa spectrum (.fits)
number of iterations needed (.fits); good to check for contamination by continuum or noisy pixels
'''


