''' spectrum utils '''

import numpy as np
from pathlib import Path
from typing import Union, Tuple, List

from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS


def pixel_circle_calculation(fitsfile : Union[Path, str],
                             glon : float,
                             glat : float,
                             r : float) -> List:
    """Extract a list of pixels [(y0,x0),(y1,x1),...] corresponding to the circle region
    with central coordinates glon, glat, and radius r.

    Parameters
    ----------
    fitsfile : Path | str
        Path to FITS file.
    glon : float
        x-coordinate of central pixel in units given in the header.
    glat : float
        y-coordinate of central pixel in units given in the header.
    r : float
        Radius of region in units of arcseconds.

    Returns
    -------
    pixel_array : List
        List of pixel coordinates [(y0,x0),(y1,x1),...].
    """
    header = fits.getheader(fitsfile)
    w = WCS(fitsfile)
    delta = abs(header['CDELT1']) #in degree
    pixel_array = []
    if header['NAXIS']==3:
        central_px = w.all_world2pix(glon,glat,0,1)
    elif header['NAXIS']==2:
        central_px = w.all_world2pix(glon,glat,1)
    else:
        raise Exception('Something wrong with the header!')
    central_px = [int(np.round(central_px[0],decimals=0))-1,int(np.round(central_px[1],decimals=0))-1]
    if r is not 'single':
        circle_size_px = 2*r/3600. / delta
        circle_size_px = int(round(circle_size_px))
        px_start = [central_px[0]-circle_size_px/2,central_px[1]-circle_size_px/2]
        px_end = [central_px[0]+circle_size_px/2,central_px[1]+circle_size_px/2]
        px_start = [int(np.round(px_start[0],decimals=0)),int(np.round(px_start[1],decimals=0))]
        px_end = [int(np.round(px_end[0],decimals=0)),int(np.round(px_end[1],decimals=0))]
        for i_x in range(px_start[0]-1,px_end[0]+1):
            for i_y in range(px_start[1]-1,px_end[1]+1):
                if np.sqrt((i_x-central_px[0])**2+(i_y-central_px[1])**2) < circle_size_px/2.:
                    pixel_array.append((i_x,i_y))
    else:
        pixel_array.append((central_px[0],central_px[1]))
    return pixel_array



def pixel_circle_calculation_px(fitsfile : Union[Path, str],
                                x : float,
                                y : float,
                                r : float) -> List:
    """Extract a list of pixels [(y0,x0),(y1,x1),...] corresponding to the circle region
    with central pixels x, y, and radius r.

    Parameters
    ----------
    fitsfile : Path | str
        Path to FITS file.
    x : float
        Central x-pixel.
    y : float
        Central y-pixel.
    r : float
        Radius of region in units of arcseconds.

    Returns
    -------
    pixel_array : List
        List of pixel coordinates [(y0,x0),(y1,x1),...].
    """
    header = fits.getheader(fitsfile)
    w = WCS(fitsfile)
    delta = abs(header['CDELT1']) #in degree
    pixel_array = []
    central_px = [x,y]
    if r is not 'single':
        circle_size_px = 2*r/3600. / delta
        circle_size_px = int(round(circle_size_px))
        px_start = [central_px[0]-circle_size_px/2,central_px[1]-circle_size_px/2]
        px_end = [central_px[0]+circle_size_px/2,central_px[1]+circle_size_px/2]
        px_start = [int(np.round(px_start[0],decimals=0)),int(np.round(px_start[1],decimals=0))]
        px_end = [int(np.round(px_end[0],decimals=0)),int(np.round(px_end[1],decimals=0))]
        for i_x in range(px_start[0]-1,px_end[0]+1):
            for i_y in range(px_start[1]-1,px_end[1]+1):
                if np.sqrt((i_x-central_px[0])**2+(i_y-central_px[1])**2) < circle_size_px/2.:
                    pixel_array.append((i_x,i_y))
    else:
        pixel_array.append((central_px[0],central_px[1]))
    return pixel_array



def calculate_spectrum(fitsfile : Union[Path, str], pixel_array : List) -> np.ndarray:
    """Calculate an average spectrum given a p-p-v FITS cube and pixel coordinates.
    If NaN values are present at specific coordinates, these coordinates will be ignored.

    Parameters
    ----------
    fitsfile : Path | str
        Path to FITS file to get average spectrum from.
    pixel_array : List
        List of tuples containing pixel coordinates [(y0,x0),(y1,x1),...]
	    over which to average.

    Returns
    -------
    spectrum_average : numpy.ndarray
        Averaged spectrum.
    """
    header = fits.getheader(fitsfile)
    image = fits.getdata(fitsfile)
    number_of_channels = header['NAXIS3']
    spectrum_add = np.zeros(number_of_channels)
    n=0
    for i in range(0,len(pixel_array)):
        x_1,y_1 = pixel_array[i]
        spectrum_i = image[:,y_1,x_1]
        if any([np.isnan(spectrum_i[i]) for i in range(len(spectrum_i))]):
            print('Warning: region contains NaNs!')
            spectrum_add = spectrum_add + 0
            n+=1
        else:
            spectrum_add = spectrum_add + spectrum_i
    spectrum_average = spectrum_add / (len(pixel_array)-n)
    return spectrum_average
