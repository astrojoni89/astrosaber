''' plotting functions '''

# @Author: syed
# @Date:   2021-03-01
# @Filename: plotting.py
# @Last modified by:   syed
# @Last modified time: 03-03-2021

import random
import numpy as np

#import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy import units as u
from tqdm import tqdm

from .utils.spectrum_utils import pixel_circle_calculation, pixel_circle_calculation_px, calculate_spectrum
from .utils.aslsq_helper import find_nearest, velocity_axes



def get_figure_params(n_spectra, cols, rowsize, rowbreak):
    colsize = rowsize
    cols = int(np.sqrt(n_spectra))
    rows = int(n_spectra / (cols))
    if n_spectra % cols != 0:
        rows += 1
    if rows < rowbreak:
        rowbreak = rows
    if (rowbreak*rowsize*100 > 2**16) or (cols*colsize*100 > 2**16):
        errorMessage = \
            "Image size is too large. It must be less than 2^16 pixels in each direction. Restrict the number of columns or rows."
        raise Exception(errorMessage)

    return cols, rows, rowbreak, colsize




def plot_spectra(fitsfiles, coordinates=None, radius=None, path_to_plots=None, n_spectra=9, cols=3, rowsize=7.75, rowbreak=50, dpi=50, velocity_range=[-110,163]):
    '''
    fitsfiles: list of fitsfiles to plot spectra from
    coordinates: array of central coordinates [[Glon, Glat]] to plot spectra from
    radius: radius of area to be averaged for each spectrum [arcseconds]
    ''' 
    if coordinates is not None:
        n_spectra = len(coordinates)
        cols, rows, rowbreak, colsize = get_figure_params(n_spectra, cols, rowsize, rowbreak)
        figsize = (cols*colsize, rowbreak*rowsize)
        fig = plt.figure(figsize=figsize)
        
        if radius is not None:
            for i in range(len(coordinates)):
                ax = fig.add_subplot(rows,cols,i+1)
                for fitsfile in fitsfiles:
                    pixel_array = pixel_circle_calculation(fitsfile,glon=coordinates[i,0],glat=coordinates[i,1],r=radius)
                    spectrum = calculate_spectrum(fitsfile,pixel_array)
                    velocity = velocity_axes(fitsfile)
                    velo_min, velo_max = find_nearest(velocity,velocity_range[np.argmin(velocity_range)]), find_nearest(velocity,velocity_range[np.argmax(velocity_range)])
                    ax.plot(velocity[velo_min:velo_max], spectrum[velo_min:velo_max])


        else:
            for i in range(len(coordinates)):
                ax = fig.add_subplot(rows,cols,i+1)
                for fitsfile in fitsfiles:
                    header = fits.getheader(fitsfile)
                    beam = header['BMAJ']
                    radius = 1/2. * (beam*3600)
                    pixel_array = pixel_circle_calculation(fitsfile,glon=coordinates[i,0],glat=coordinates[i,1],r=radius)
                    spectrum = calculate_spectrum(fitsfile,pixel_array)
                    velocity = velocity_axes(fitsfile)
                    velo_min, velo_max = find_nearest(velocity,velocity_range[np.argmin(velocity_range)]), find_nearest(velocity,velocity_range[np.argmax(velocity_range)])
                    ax.plot(velocity[velo_min:velo_max], spectrum[velo_min:velo_max])

    else:
        random.seed(111)
        edge = 20
        xsize = fits.getdata(fitsfiles[0]).shape[2]
        ysize = fits.getdata(fitsfiles[0]).shape[1]
        cols, rows, rowbreak, colsize = get_figure_params(n_spectra, cols, rowsize, rowbreak)
        figsize = (cols*colsize, rowbreak*rowsize)
        fig = plt.figure(figsize=figsize)

        if radius is not None:
            for i in range(n_spectra):
                xValue = random.randint(edge,xsize-edge)
                yValue = random.randint(edge,ysize-edge)
                ax = fig.add_subplot(rows,cols,i+1)
                for fitsfile in fitsfiles:
                    pixel_array = pixel_circle_calculation_px(fitsfile,x=xValue,y=yValue,r=radius)
                    spectrum = calculate_spectrum(fitsfile,pixel_array)
                    velocity = velocity_axes(fitsfile)
                    velo_min, velo_max = find_nearest(velocity,velocity_range[np.argmin(velocity_range)]), find_nearest(velocity,velocity_range[np.argmax(velocity_range)])
                    ax.plot(velocity[velo_min:velo_max], spectrum[velo_min:velo_max])


        else:
            for i in range(n_spectra):
                xValue = random.randint(edge,xsize-edge)
                yValue = random.randint(edge,ysize-edge)
                ax = fig.add_subplot(rows,cols,i+1)
                for fitsfile in fitsfiles:
                    header = fits.getheader(fitsfile)
                    beam = header['BMAJ']
                    radius = 1/2. * (beam*3600)
                    pixel_array = pixel_circle_calculation_px(fitsfile,x=xValue,y=yValue,r=radius)
                    spectrum = calculate_spectrum(fitsfile,pixel_array)
                    velocity = velocity_axes(fitsfile)
                    velo_min, velo_max = find_nearest(velocity,velocity_range[np.argmin(velocity_range)]), find_nearest(velocity,velocity_range[np.argmax(velocity_range)])
                    ax.plot(velocity[velo_min:velo_max], spectrum[velo_min:velo_max])


