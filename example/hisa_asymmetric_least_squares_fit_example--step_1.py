import os
import numpy as np
from astropy.io import fits
from astropy import units as u

from astrosaber.prepare_training import saberPrepare


###step 1: create training data that are generated from 'pure' emission spectra (test_data) with randomly generated (but known) self-absorption features


def main():
    ###HI data to extract training data from
    filename = 'HI_THOR_test_cube.fits'

    ###initialize training set preparation
    prep = saberPrepare(fitsfile=filename)

    ###set the size of the training set
    prep.training_set_size = 50

    ###path to noise map (or universal noise value)
    #prep.path_to_noise_map = os.path.join('.', 'dir', 'sub', '*.fits')
    prep.noise = 4. # Kelvin

    ###set the expected linewidth of self-absorption features; artificial self-absorption features will be generated from this distribution
    prep.mean_linewidth = 4. # FWHM [km/s]
    prep.std_linewidth = 1. # standard deviation of the linewidth distribution [km/s]

    ###you can always print the prep object and the keyword arguments you can adjust
    print(prep)

    ###this runs the training set extraction
    prep.prepare_training()

    '''
    the output will be one .pickle file saved in the 'astrosaber_training' directory.
    It contains the training_data (i.e., emission spectra with self-absorption),
    test_data ('real' emission spectra that is to be recovered),
    and header information
    '''


if __name__ == '__main__':
    main()
