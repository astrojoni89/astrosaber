''' plotting functions '''

# @Author: syed
# @Date:   2021-03-01
# @Filename: plotting.py
# @Last modified by:   syed
# @Last modified time: 03-03-2021

import os
import random
import numpy as np

import matplotlib.pyplot as plt

from astropy.io import fits
from astropy import units as u
from tqdm import tqdm

from .utils.spectrum_utils import pixel_circle_calculation, pixel_circle_calculation_px, calculate_spectrum
from .utils.aslsq_helper import find_nearest, velocity_axes

def styles():
    color_list = ['k', 'r', 'r']
    draw_list = ['steps-mid', 'default', 'steps-mid']
    line_list = ['-', '--', '-']
    return color_list, draw_list, line_list

def get_figure_params(n_spectra, rowsize, rowbreak):
    colsize = ((1+np.sqrt(5))/2) * rowsize
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


def xlabel_from_header(header, vel_unit):
    xlabel = 'Channels'

    if header is None:
        return xlabel

    if 'CTYPE3' in header.keys():
        xlabel = '{} [{}]'.format(header['CTYPE3'], vel_unit)

    return xlabel


def ylabel_from_header(header):
    if header is None:
        return 'Intensity'

    btype = 'Intensity'
    if 'BTYPE' in header.keys():
        btype = header['BTYPE']

    bunit = ''
    if 'BUNIT' in header.keys():
        bunit = ' [{}]'.format(header['BUNIT'])

    return btype + bunit


def add_figure_properties(ax, header=None, fontsize=10, vel_unit=u.km/u.s):
    ax.set_xlabel(xlabel_from_header(header, vel_unit), fontsize=fontsize)
    ax.set_ylabel(ylabel_from_header(header), fontsize=fontsize)

    ax.tick_params(labelsize=fontsize - 2)

    
def scale_fontsize(rowsize):
    rowsize_scale = 4
    if rowsize >= rowsize_scale:
        fontsize = 10 + int(rowsize - rowsize_scale)
    else:
        fontsize = 10 - int(rowsize - rowsize_scale)
    return fontsize


def plot_spectra(fitsfiles, outfile='spectra.pdf', coordinates=None, radius=None, path_to_plots='.', n_spectra=9, rowsize=4., rowbreak=10, dpi=72, velocity_range=[-110,163], vel_unit=u.km/u.s):
    '''
    fitsfiles: list of fitsfiles to plot spectra from
    coordinates: array of central coordinates [[Glon, Glat]] to plot spectra from
    radius: radius of area to be averaged for each spectrum [arcseconds]
    '''
    
    print("\nPlotting...")
    
    fontsize = scale_fontsize(rowsize)
    if len(fitsfiles==3):
        color_list, draw_list, line_list = styles()
    
    if coordinates is not None:
        n_spectra = len(coordinates)
        cols, rows, rowbreak, colsize = get_figure_params(n_spectra, rowsize, rowbreak)
        figsize = (cols*colsize, rowbreak*rowsize)
        fig = plt.figure(figsize=figsize)
        
        if radius is not None:
            for i in range(len(coordinates)):
                ax = fig.add_subplot(rows,cols,i+1)
                for idx, fitsfile in enumerate(fitsfiles):
                    pixel_array = pixel_circle_calculation(fitsfile,glon=coordinates[i,0],glat=coordinates[i,1],r=radius)
                    spectrum = calculate_spectrum(fitsfile,pixel_array)
                    header = fits.getheader(fitsfile)
                    velocity = velocity_axes(fitsfile)
                    velo_min, velo_max = find_nearest(velocity,np.amin(velocity_range)), find_nearest(velocity,np.amax(velocity_range))
                    ax.plot(velocity[velo_min:velo_max], spectrum[velo_min:velo_max], drawstyle=draw_list[idx], color=color_list[idx], linestyle=line_list[idx])
                add_figure_properties(ax, header=header, fontsize=fontsize, vel_unit=vel_unit)
                

        else:
            for i in range(len(coordinates)):
                ax = fig.add_subplot(rows,cols,i+1)
                for idx, fitsfile in enumerate(fitsfiles):
                    header = fits.getheader(fitsfile)
                    beam = header['BMAJ']
                    radius = 1/2. * (beam*3600)
                    pixel_array = pixel_circle_calculation(fitsfile,glon=coordinates[i,0],glat=coordinates[i,1],r=radius)
                    spectrum = calculate_spectrum(fitsfile,pixel_array)
                    velocity = velocity_axes(fitsfile)
                    velo_min, velo_max = find_nearest(velocity,np.amin(velocity_range)), find_nearest(velocity,np.amax(velocity_range))
                    ax.plot(velocity[velo_min:velo_max], spectrum[velo_min:velo_max], drawstyle=draw_list[idx], color=color_list[idx], linestyle=line_list[idx])
                add_figure_properties(ax, header=header, fontsize=fontsize, vel_unit=vel_unit)

    else:
        random.seed(111)
        edge = 20
        xsize = fits.getdata(fitsfiles[0]).shape[2]
        ysize = fits.getdata(fitsfiles[0]).shape[1]
        cols, rows, rowbreak, colsize = get_figure_params(n_spectra, rowsize, rowbreak)
        figsize = (cols*colsize, rowbreak*rowsize)
        fig = plt.figure(figsize=figsize)

        if radius is not None:
            for i in range(n_spectra):
                xValue = random.randint(edge,xsize-edge)
                yValue = random.randint(edge,ysize-edge)
                ax = fig.add_subplot(rows,cols,i+1)
                for idx, fitsfile in enumerate(fitsfiles):
                    pixel_array = pixel_circle_calculation_px(fitsfile,x=xValue,y=yValue,r=radius)
                    spectrum = calculate_spectrum(fitsfile,pixel_array)
                    header = fits.getheader(fitsfile)
                    velocity = velocity_axes(fitsfile)
                    velo_min, velo_max = find_nearest(velocity,np.amin(velocity_range)), find_nearest(velocity,np.amax(velocity_range))
                    ax.plot(velocity[velo_min:velo_max], spectrum[velo_min:velo_max], drawstyle=draw_list[idx], color=color_list[idx], linestyle=line_list[idx])
                add_figure_properties(ax, header=header, fontsize=fontsize, vel_unit=vel_unit)

        else:
            for i in range(n_spectra):
                xValue = random.randint(edge,xsize-edge)
                yValue = random.randint(edge,ysize-edge)
                ax = fig.add_subplot(rows,cols,i+1)
                for idx, fitsfile in enumerate(fitsfiles):
                    header = fits.getheader(fitsfile)
                    beam = header['BMAJ']
                    radius = 1/2. * (beam*3600)
                    pixel_array = pixel_circle_calculation_px(fitsfile,x=xValue,y=yValue,r=radius)
                    spectrum = calculate_spectrum(fitsfile,pixel_array)
                    velocity = velocity_axes(fitsfile)
                    velo_min, velo_max = find_nearest(velocity,np.amin(velocity_range)), find_nearest(velocity,np.amax(velocity_range))
                    ax.plot(velocity[velo_min:velo_max], spectrum[velo_min:velo_max], drawstyle=draw_list[idx], color=color_list[idx], linestyle=line_list[idx])
                add_figure_properties(ax, header=header, fontsize=fontsize, vel_unit=vel_unit)

    for axs in fig.axes:
        axs.label_outer()
    fig.tight_layout()

    if not os.path.exists(path_to_plots):
        os.makedirs(path_to_plots)
    filename = outfile
    pathname = os.path.join(path_to_plots, filename)
    fig.savefig(pathname, dpi=dpi, bbox_inches='tight')
    #plt.close()
    print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(filename, path_to_plots))
