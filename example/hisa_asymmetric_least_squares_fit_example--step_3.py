import os
import numpy as np
from astropy.io import fits
from astropy import units as u

from astrosaber.hisa import HisaExtraction
from astrosaber.plotting import plot_spectra


###step 3: run the self-absorption extraction with optimal smoothing parameters obtained in step 2


def main():
    ###HI data to extract HISA
    image_HI = 'HI_THOR_test_cube.fits'

    ###initialize hisa extraction
    hisa = HisaExtraction(fitsfile=image_HI)

    ###path to noise map (or universal noise value)
    #hisa.path_to_noise_map = os.path.join('.', 'dir', 'sub', '*.fits')
    hisa.noise = 4. #Kelvin

    ###put optimal asymmetric least squares smoothing parameters here
    hisa.lam1 = 3.469
    hisa.lam2 = 0.555

    ###maximum number of iterations (this limit is usually reached for strong continuum sources)
    hisa.niters = 20

    ###this runs the hisa extraction routine

    hisa.ncpus = 4
    hisa.baby_yoda = False
    hisa.saber()

    '''
    the output will be four (three if output_flags = False) files:
    hisa background spectrum (.fits)
    hisa spectrum (.fits)
    number of iterations needed (.fits); good to check for contamination by continuum or noisy pixels
    (optional) map of flags (.fits); 1: good pixels, 0: flagged spectra that did not meet convergence criteria or were discarded due to missing signal
    '''

    ###plot nine example spectra at random positions; or give it some coordinates as array; by default the spectra are averaged over one beam size
    coords = np.loadtxt('coords.txt') #to plot spectra at these positions: plot_spectra(coordinates=coords)
    fitsfiles = ['HI_THOR_test_cube.fits', 'HI_THOR_test_cube_aslsq_bg_spectrum.fits', 'HI_THOR_test_cube_HISA_spectrum.fits']
    plot_spectra(fitsfiles, outfile='spectra_astroSABER.pdf', coordinates=None, n_spectra=9, velocity_range=[-110,163])


if __name__ == '__main__':
    main()
