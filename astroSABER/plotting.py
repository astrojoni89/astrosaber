''' plotting functions '''

# @Author: syed
# @Date:   2021-03-01
# @Filename: plotting.py
# @Last modified by:   syed
# @Last modified time: 07-12-2021

import os
import sys
import numpy as np
import pickle

import matplotlib.pyplot as plt

from astropy.io import fits
from astropy import units as u
from tqdm import trange

from .utils.spectrum_utils import pixel_circle_calculation, pixel_circle_calculation_px, calculate_spectrum
from .utils.aslsq_helper import find_nearest, velocity_axes, pixel_to_world


def pickle_load_file(pathToFile):
    with open(os.path.join(pathToFile), "rb") as pickle_file:
        if (sys.version_info > (3, 0)):
            data = pickle.load(pickle_file, encoding='latin1')
        else:
            data = pickle.load(pickle_file)
    return data

def styles():
    color_list = ['k', 'b', 'b', 'r', 'g']
    draw_list = ['steps-mid', 'default', 'steps-mid', 'steps-mid', 'steps-mid']
    line_list = ['-', '--', '-', '-', '-']
    return color_list, draw_list, line_list

def styles_pickle():
    color_list = ['b', 'k', 'r', 'r', 'g']
    draw_list = ['steps-mid', 'steps-mid', 'default', 'steps-mid', 'steps-mid']
    line_list = ['-', '-', '-', '-', '-']
    return color_list, draw_list, line_list

def get_figure_params(n_spectra, rowsize, rowbreak):
    colsize = 1.3 * rowsize
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

def add_figure_properties(ax, header=None, fontsize=10, velocity_range=None, vel_unit=u.km/u.s):
    ax.set_xlim(np.amin(velocity_range), np.amax(velocity_range))
    #ax.set_ylim()
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

def plot_spectra(fitsfiles, outfile='spectra.pdf', coordinates=None, radius=None, path_to_plots='.', n_spectra=9, rowsize=4., rowbreak=10, dpi=72, velocity_range=[-110,163], vel_unit=u.km/u.s, seed=111):
    '''
    fitsfiles: list of fitsfiles to plot spectra from
    coordinates: array of central coordinates [[Glon, Glat]] to plot spectra from
    radius: radius of area to be averaged for each spectrum [arcseconds]
    '''
    
    print("\nPlotting...")
    
    fontsize = scale_fontsize(rowsize)
    color_list, draw_list, line_list = styles()
    
    if coordinates is not None:
        n_spectra = len(coordinates)
        cols, rows, rowbreak, colsize = get_figure_params(n_spectra, rowsize, rowbreak)
        figsize = (cols*colsize, rowbreak*rowsize)
        fig = plt.figure(figsize=figsize)
        
        if radius is not None:
            for i in trange(len(coordinates)):
                ax = fig.add_subplot(rows,cols,i+1)
                for idx, fitsfile in enumerate(fitsfiles):
                    pixel_array = pixel_circle_calculation(fitsfile,glon=coordinates[i,0],glat=coordinates[i,1],r=radius)
                    spectrum = calculate_spectrum(fitsfile,pixel_array)
                    header = fits.getheader(fitsfile)
                    velocity = velocity_axes(fitsfile)
                    velo_min, velo_max = find_nearest(velocity,np.amin(velocity_range)), find_nearest(velocity,np.amax(velocity_range))
                    ax.plot(velocity[velo_min:velo_max], spectrum[velo_min:velo_max], drawstyle=draw_list[idx], color=color_list[idx], linestyle=line_list[idx])
                add_figure_properties(ax, header=header, fontsize=fontsize, velocity_range=velocity_range, vel_unit=vel_unit)
                plt.annotate('Glon: {} deg\nGlat: {} deg'.format(round(coordinates[i][0],2),round(coordinates[i][1],2)), xy=(0.05, 0.85), xycoords='axes fraction', fontsize=fontsize)
                

        else:
            for i in trange(len(coordinates)):
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
                add_figure_properties(ax, header=header, fontsize=fontsize, velocity_range=velocity_range, vel_unit=vel_unit)
                plt.annotate('Glon: {} deg\nGlat: {} deg'.format(round(coordinates[i][0],2),round(coordinates[i][1],2)), xy=(0.05, 0.85), xycoords='axes fraction', fontsize=fontsize)

    else:
        rng = np.random.default_rng(seed)
        xsize = fits.getdata(fitsfiles[0]).shape[2]
        ysize = fits.getdata(fitsfiles[0]).shape[1]
        cols, rows, rowbreak, colsize = get_figure_params(n_spectra, rowsize, rowbreak)
        figsize = (cols*colsize, rowbreak*rowsize)
        fig = plt.figure(figsize=figsize)

        if radius is not None:
            for i in trange(n_spectra):
                temp_header = fits.getheader(fitsfiles[0])
                px_scale = abs(temp_header['CDELT1'])
                edge = int(np.ceil((radius/3600) / px_scale))
                xValue = rng.integers(edge+1,xsize-edge)
                yValue = rng.integers(edge+1,ysize-edge)
                ax = fig.add_subplot(rows,cols,i+1)
                for idx, fitsfile in enumerate(fitsfiles):
                    pixel_array = pixel_circle_calculation_px(fitsfile,x=xValue,y=yValue,r=radius)
                    spectrum = calculate_spectrum(fitsfile,pixel_array)
                    header = fits.getheader(fitsfile)
                    velocity = velocity_axes(fitsfile)
                    velo_min, velo_max = find_nearest(velocity,np.amin(velocity_range)), find_nearest(velocity,np.amax(velocity_range))
                    ax.plot(velocity[velo_min:velo_max], spectrum[velo_min:velo_max], drawstyle=draw_list[idx], color=color_list[idx], linestyle=line_list[idx])
                add_figure_properties(ax, header=header, fontsize=fontsize, velocity_range=velocity_range, vel_unit=vel_unit)
                coordinate = pixel_to_world(fitsfiles[0],xValue,yValue)
                plt.annotate('Glon: {} deg\nGlat: {} deg'.format(round(coordinate[0].item(0),2),round(coordinate[1].item(0),2)), xy=(0.05, 0.85), xycoords='axes fraction', fontsize=fontsize)

        else:
            for i in trange(n_spectra):
                temp_header = fits.getheader(fitsfiles[0])
                px_scale = abs(temp_header['CDELT1'])
                temp_beam = temp_header['BMAJ']
                temp_radius = 1/2. * temp_beam
                edge = int(np.ceil(temp_radius / px_scale))
                xValue = rng.integers(edge+1,xsize-edge)
                yValue = rng.integers(edge+1,ysize-edge)
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
                add_figure_properties(ax, header=header, fontsize=fontsize, velocity_range=velocity_range, vel_unit=vel_unit)
                coordinate = pixel_to_world(fitsfiles[0],xValue,yValue)
                plt.annotate('Glon: {} deg\nGlat: {} deg'.format(round(coordinate[0].item(0),2),round(coordinate[1].item(0),2)), xy=(0.05, 0.85), xycoords='axes fraction', fontsize=fontsize)

    #for axs in fig.axes:
        #axs.label_outer()
    fig.tight_layout()

    if not os.path.exists(path_to_plots):
        os.makedirs(path_to_plots)
    filename = outfile
    pathname = os.path.join(path_to_plots, filename)
    fig.savefig(pathname, dpi=dpi, bbox_inches='tight')
    #plt.close()
    print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(filename, path_to_plots))
      
def plot_pickle_spectra(pickle_file, outfile='spectra.pdf', ranges=None, path_to_plots='.', n_spectra=9, rowsize=4., rowbreak=10, dpi=72, velocity_range=[-110,163], vel_unit=u.km/u.s, seed=111):
    '''
    pickle_file: pickled file to plot spectra from
    '''
    
    print("\nPlotting...")
    
    fontsize = scale_fontsize(rowsize)
    color_list, draw_list, line_list = styles_pickle()

    data = pickle_load_file(pickle_file)
    training_data = data['training_data']
    test_data = data['test_data']
    velocity = data['velocity']
    if 'header' in data.keys():
        header = data['header']
    else:
        header = None
        
    rng = np.random.default_rng(seed)
    xsize = len(data['training_data'])
    cols, rows, rowbreak, colsize = get_figure_params(n_spectra, rowsize, rowbreak)
    figsize = (cols*colsize, rowbreak*rowsize)
    fig = plt.figure(figsize=figsize)
    xValue = rng.integers(0,high=xsize,size=n_spectra)
    for i in trange(n_spectra):
        idx = xValue[i]
        ax = fig.add_subplot(rows,cols,i+1)
        velo_min, velo_max = find_nearest(velocity,np.amin(velocity_range)), find_nearest(velocity,np.amax(velocity_range))
        ax.plot(velocity[velo_min:velo_max], test_data[idx][velo_min:velo_max], drawstyle=draw_list[0], color=color_list[0], linestyle=line_list[0], label="'pure' HI")
        ax.plot(velocity[velo_min:velo_max], training_data[idx][velo_min:velo_max], drawstyle=draw_list[1], color=color_list[1], linestyle=line_list[1], label="observed HI+HISA")
        add_figure_properties(ax, header=header, fontsize=fontsize, velocity_range=velocity_range, vel_unit=vel_unit)
        ax.legend(loc=2, fontsize=fontsize-2)

    #for axs in fig.axes:
        #axs.label_outer()
    fig.tight_layout()

    if not os.path.exists(path_to_plots):
        os.makedirs(path_to_plots)
    if outfile is not None:
        filename = outfile
    elif outfile is None:
        filename = pickle_file.split('/')[-1].split('.pickle')[0] + '_{}.pdf'.format(n_spectra)
    pathname = os.path.join(path_to_plots, filename)
    fig.savefig(pathname, dpi=dpi, bbox_inches='tight')
    #plt.close()
    print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(filename, path_to_plots))

def plot_training_spectra(pickle_file, training_data, test_data, bg_fits, outfile='spectra.pdf', ranges=None, path_to_plots='.', n_spectra=9, rowsize=4., rowbreak=10, dpi=72, velocity_range=[-110,163], vel_unit=u.km/u.s, seed=111):
    '''
    pickle_file: pickled file to get x-axis from
    '''
    
    print("\nPlotting...")
    
    fontsize = scale_fontsize(rowsize)
    color_list, draw_list, line_list = styles_pickle()

    data = pickle_load_file(pickle_file)
    training_data = training_data
    test_data = test_data
    bg_fits = bg_fits
    velocity = data['velocity']
    if 'header' in data.keys():
        header = data['header']
    else:
        header = None
        
    rng = np.random.default_rng(seed)
    xsize = len(training_data)
    cols, rows, rowbreak, colsize = get_figure_params(n_spectra, rowsize, rowbreak)
    figsize = (cols*colsize, rowbreak*rowsize)
    fig = plt.figure(figsize=figsize)
    xValue = rng.integers(0,high=xsize,size=n_spectra)
    for i in trange(n_spectra):
        idx = xValue[i]
        ax = fig.add_subplot(rows,cols,i+1)
        velo_min, velo_max = find_nearest(velocity,np.amin(velocity_range)), find_nearest(velocity,np.amax(velocity_range))
        ax.plot(velocity[velo_min:velo_max], test_data[idx][velo_min:velo_max], drawstyle=draw_list[0], color=color_list[0], linestyle=line_list[0], label="'pure' HI")
        ax.plot(velocity[velo_min:velo_max], training_data[idx][velo_min:velo_max], drawstyle=draw_list[1], color=color_list[1], linestyle=line_list[1], label="observed HI+HISA")
        ax.plot(velocity[velo_min:velo_max], bg_fits[idx][velo_min:velo_max], drawstyle=draw_list[2], color=color_list[2], linestyle=line_list[2], label="bg fit")
        add_figure_properties(ax, header=header, fontsize=fontsize, velocity_range=velocity_range, vel_unit=vel_unit)
        ax.legend(loc=2, fontsize=fontsize-2)

    #for axs in fig.axes:
        #axs.label_outer()
    fig.tight_layout()

    if not os.path.exists(path_to_plots):
        os.makedirs(path_to_plots)
    if outfile is not None:
        filename = outfile
    elif outfile is None:
        filename = pickle_file.split('/')[-1].split('.pickle')[0] + '_astrosaber_fits_{}.pdf'.format(n_spectra)
    pathname = os.path.join(path_to_plots, filename)
    fig.savefig(pathname, dpi=dpi, bbox_inches='tight')
    #plt.close()
    print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(filename, path_to_plots))
