'''helper functions'''

import os
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from scipy import sparse
from scipy.sparse.linalg import spsolve

from tqdm import trange
import warnings
import sys


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


def velocity_axes(name):
	header = fits.getheader(name)
	n = header['NAXIS3']
	velocity = (header['CRVAL3'] - header['CRPIX3'] * header['CDELT3']) + (np.arange(n)+1) * header['CDELT3']
	velocity = velocity / 1000
	return velocity


def pixel_to_world(fitsfile,x,y,ch=0):
    try:
        w = WCS(fitsfile)
        if w.wcs.naxis == 3:
            return w.all_pix2world(x, y, ch, 1)
        elif w.wcs.naxis == 2:
            return w.all_pix2world(x, y, 1)
        else:
            raise ValueError('Something wrong with the header.')
    except:
        return [np.array([x]), np.array([y])]


#taken from Lindner (2014) & Riener (2019); GaussPy(+)
def count_ones_in_row(data):
    """ Counts number of continuous trailing '1's
         Used in the convergence criteria
    """
    output = np.zeros(len(data))
    for i in range(len(output)):
        if data[i] == 0:
            output[i] = 0
        else:
            total = 1
            current = 1
            counter = 1
            while data[i-counter] == 1:
                total += 1
                if i - counter < 0:
                    break
                current = data[i - counter]
                counter += 1
            output[i] = total
    return output


#simple check
def check_signal(spectrum, sigma, noise):
    return np.any(spectrum > sigma * noise)
            

#check if there is signal in at least xy neighboring channels corresponding to velo_range [km/s]: default 10 channels
def check_signal_ranges(spectrum, header, sigma=None, noise=None, velo_range=None):
    vdelt = header['CDELT3'] / 1000.
    if sigma is None:
        sigma = 5
    if noise is None:
        noise = np.round(np.nanstd(np.append(spectrum[0:int(10*vdelt)],spectrum[int(-10*vdelt):-1])),decimals=2)
    if np.any(spectrum > sigma * noise):
        if velo_range is not None:
            min_channels = velo_range//vdelt
        else:
            min_channels = 10
        channel_logic = np.array([])
        for i in range(len(spectrum)):
            channel_test = np.all(spectrum[i] > sigma*noise) 
            channel_logic = np.append(channel_logic,channel_test)
        c = count_ones_in_row(channel_logic)
        return np.any(c > min_channels)
    else:
        return False


def md_header_2d(fitsfile):
    header_2d = fits.getheader(fitsfile)
    del header_2d['NAXIS3']
    del header_2d['CRPIX3']
    del header_2d['CDELT3']
    del header_2d['CUNIT3']
    del header_2d['CTYPE3']
    del header_2d['CRVAL3']

    header_2d['NAXIS'] = 2
    header_2d['WCSAXES'] = 2
    return header_2d


class IterationWarning(UserWarning):
    pass


def say(message, verbose=True, end=None):
    """Diagnostic messages."""
    if verbose:
        print(message, end=end)


def format_warning(message, category, filename, lineno, file=None, line=None):
    sys.stderr.write("\n\033[93mWARNING:\033[0m {}: {}\n".format(category.__name__, message))
