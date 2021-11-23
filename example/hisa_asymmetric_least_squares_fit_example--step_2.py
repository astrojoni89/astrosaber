import os
import numpy as np
from astropy.io import fits
from astropy import units as u

from astroSABER.training import saberTraining


###step 2: use training and test data obtained in step 1 to find optimal smoothing parameters


def main():
    ###data to use as training and test data
    filename = 'astrosaber_training/.pickle'

    ###initialize optimization routine
    train = saberTraining(pickle_file=filename)

    ###you can adjust the number of cpus to use
    train.ncpus = 4

    ###set the initial guesses for the smoothing parameters (better to start low rather than high)
    train.lam1_initial = 2.00
    train.lam2_initial = 1.00

    ###you can always print the train object and the keyword arguments you can adjust
    print(train)

    ###this runs the optimization routine
    train.training()

    '''
    the output will be one .txt file containing the two optimal smoothing parameters
    '''


if __name__ == '__main__':
    main()
