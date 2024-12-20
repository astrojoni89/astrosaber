'''helper functions'''

import numpy as np
from pathlib import Path
from typing import Union, Optional

from astropy.io import fits
from astropy.wcs import WCS

import sys


def find_nearest(array: np.ndarray, value: float) -> int:
    """
    Find the index of an element in an array nearest to a given value.

    Parameters
    ----------
    array : numpy.ndarray
        Input array to index.
    value : float
        Value of the element to find the closest index for.

    Returns
    -------
    idx : int
        Index of the element with value closest to `value`.
    """
    idx = (np.abs(array-value)).argmin()
    return idx


def velocity_axes(name: Path, cunit3: str = 'm/s') -> np.ndarray:
    """
    Get velocity axis from FITS file in units of km/s.

    Parameters
    ----------
    name : Path
        Path to FITS file to get velocity axis from.
    cunit3 : str, optional
        Type of velocity unit specified in the fits file header keyword 'CUNIT3'.
        Default is 'm/s'.

    Returns
    -------
    velocity : numpy.ndarray
        Array of velocity axis.
    """
    header = fits.getheader(name)
    n = header['NAXIS3']
    velocity = (header['CRVAL3'] - header['CRPIX3'] * header['CDELT3']) + (np.arange(n)+1) * header['CDELT3']
    if cunit3 == 'm/s':
        velocity = velocity / 1000
    elif cunit3 == 'km/s':
        return velocity
    else:
        raise ValueError('Unknown velocity unit (cunit3)')
    return velocity


def merge_ranges(ranges : np.ndarray) -> np.ndarray:
    """
    Merge intervals where they overlap.

    Parameters
    ----------
    ranges : numpy.ndarray
        Array of signal intervals indicating the start and end index.

    Returns
    -------
    merged_ranges : numpy.ndarray
        Array of merged ranges.
    """
    list(ranges).sort(key=lambda interval: interval[0])
    merged_ranges = [ranges[0]]
    for current in ranges:
        previous = merged_ranges[-1]
        if current[0] <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            merged_ranges.append(current)
    return np.array(merged_ranges)


def pixel_to_world(fitsfile: Path, x: float,
                   y: float, ch: Optional[float] = 0.):
    """
    Convert pixel coordinates to world coordinates from a FITS file.

    Parameters
    ----------
    fitsfile : str
        Path to FITS file to get coordinates from.
    x : float
        Pixel coordinate on the x-axis of the FITS file.
    y : float
        Pixel coordinate on the y-axis of the FITS file.
    ch : float, optional
        Velocity channel to convert (default is 0.).

    Returns
    -------
    result : numpy.ndarray
        Returns the world coordinates. If the input was a single array and origin,
	    a single array is returned, otherwise a tuple of arrays is returned.
    """
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
def count_ones_in_row(data : np.ndarray) -> np.ndarray:
    """
    Counts number of continuous trailing '1's.

    Parameters
    ----------
    data : numpy.ndarray
        Data containing 0's and 1's.

    Returns
    -------
    output : numpy.ndarray
        Array containing the number of continuous 1's to the point of each index.
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
def check_signal(spectrum : np.ndarray, sigma : float, noise : float) -> bool:
    """
    Check for signal in an array given a significance threshold.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Data to check for signal.
    sigma : float
        Significance of the signal to be checked for.
    noise : float
        Noise level. Signal threshold will be `sigma` * `noise`.

    Returns
    -------
    check : bool
        `True` if there any data point is significant, `False` if everything below threshold.
    """
    check = np.any(spectrum > sigma * noise)
    return check


#check if there is signal in at least xy neighboring channels corresponding to velo_range [km/s]: default 10 channels
def check_signal_ranges(spectrum, header, sigma=None, noise=None, velo_range=None, cunit3='m/s'):
    """
    Check for continuous signal range in an array given a significance threshold.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Data to check for signal.
    header :
        Header of the file containing the spectrum. This is required to read out the velocity resolution.
    sigma : float
        Significance of the signal to be checked for.
    noise : float
        Noise level. Signal threshold will be `sigma` * `noise`.
    velo_range : float
        Velocity range [in km/s] of the spectrum that has to contain continuous significant signal.
    cunit3 : str, optional
        Type of velocity unit specified in the fits file header keyword 'CUNIT3'.
        Default is 'm/s'.

    Returns
    -------
    check : bool
        `True` if there is continuous signal, `False` if no signal.
    """
    if header is None: # fallback to fit all spectra if no header is given
        return True
    if cunit3 == 'm/s':
        vdelt = header['CDELT3'] / 1000.
    elif cunit3 == 'km/s':
        vdelt = header['CDELT3']
    else:
        raise ValueError('Unknown velocity unit (cunit3)')
    if sigma is None:
        sigma = 5
    if noise is None:
        noise = np.round(np.nanstd(np.append(spectrum[0:int(10*vdelt)],spectrum[int(-10*vdelt):-1])),decimals=2)
    if noise==0.:
        return True
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


def md_header_2d(fitsfile : Union[Path, str]) -> fits.Header:
    """
    Get 2D header from FITS file.

    Parameters
    ----------
    fitsfile : path-like object or file-like object
        Path to FITS file to get header from.

    Returns
    -------
    header_2d : `~astropy.io.fits.Header <https://docs.astropy.org/en/stable/io/fits/api/headers.html#astropy.io.fits.Header>`__
        Header object without third axis.
    """
    header_2d = fits.getheader(fitsfile)
    if 'NAXIS3' in header_2d.keys():
        del header_2d['NAXIS3']
    if 'CRPIX3' in header_2d.keys():
        del header_2d['CRPIX3']
    if 'CDELT3' in header_2d.keys():
        del header_2d['CDELT3']
    if 'CUNIT3' in header_2d.keys():
        del header_2d['CUNIT3']
    if 'CTYPE3' in header_2d.keys():
        del header_2d['CTYPE3']
    if 'CRVAL3' in header_2d.keys():
        del header_2d['CRVAL3']

    header_2d['NAXIS'] = 2
    header_2d['WCSAXES'] = 2
    return header_2d


class IterationWarning(UserWarning):
    """
    Passing on diagnostic messages.
    """
    pass


def say(message, verbose=True, end=None):
    """
    Diagnostic messages.
    """
    if verbose:
        print(message, end=end)


def format_warning(message, category, filename, lineno, file=None, line=None):
    """
    Print warning message.
    """
    sys.stderr.write("\n\033[93mWARNING:\033[0m {}: {}\n".format(category.__name__, message))
