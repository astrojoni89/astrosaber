''' plotting functions '''

# @Author: syed
# @Date:   2021-03-01
# @Filename: plotting.py
# @Last modified by:   syed
# @Last modified time: 01-12-2024

import os
import sys
import numpy as np
import pickle

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from astropy.io import fits
from astropy import units as u
from tqdm import trange

from .utils.spectrum_utils import pixel_circle_calculation, pixel_circle_calculation_px, calculate_spectrum
from .utils.aslsq_helper import find_nearest, velocity_axes, pixel_to_world, merge_ranges


def pickle_load_file(pathToFile):
    """
    Load a pickle file.
    """
    with open(os.path.join(pathToFile), "rb") as pickle_file:
        if (sys.version_info > (3, 0)):
            data = pickle.load(pickle_file, encoding='latin1')
        else:
            data = pickle.load(pickle_file)
    return data

def styles():
    """
    Set default plotting styles.
    """
    color_list = ['k', 'r', 'r', 'b', 'g']
    draw_list = ['steps-mid', 'default', 'steps-mid', 'steps-mid', 'steps-mid']
    line_list = ['-', '-', '-', '-', '-']
    return color_list, draw_list, line_list

def styles_pickle():
    color_list = ['b', 'k', 'r', 'r', 'g']
    draw_list = ['steps-mid', 'steps-mid', 'default', 'steps-mid', 'steps-mid']
    line_list = ['-', '-', '-', '-', '-']
    return color_list, draw_list, line_list

def get_figure_params(n_spectra, rowsize, rowbreak):
    golden_ratio = (1 + np.sqrt(5)) / 2.
    colsize = golden_ratio * rowsize
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

def add_figure_properties(ax, header=None, fontsize=10, velocity_range=None, vel_unit=u.km/u.s, set_xlabel = True, set_ylabel=True):
    ax.set_xlim(np.amin(velocity_range), np.amax(velocity_range))
    #ax.set_ylim()
    if set_xlabel:
        ax.set_xlabel(xlabel_from_header(header, vel_unit), fontsize=fontsize)
    if set_ylabel:
        ax.set_ylabel(ylabel_from_header(header), fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 2)

#def autoscale_y(fig):
#    axes = fig.get_axes()
#    #determine axes and their limits 
#    ax_selec = [(ax, ax.get_ylim()) for ax in axes]
#
#    #find maximum y-limit spread
#    max_delta = max([lmax-lmin for _, (lmin, lmax) in ax_selec])
#
#    #expand limits of all subplots according to maximum spread
#    for ax, (lmin, lmax) in ax_selec:
#        ax.set_ylim(lmin-(max_delta-(lmax-lmin))/2, lmax+(max_delta-(lmax-lmin))/2)
    

def scale_fontsize(rowsize):
    rowsize_scale = 4
    if rowsize >= rowsize_scale:
        fontsize = 10 + int(rowsize - rowsize_scale)
    else:
        fontsize = 10 - int(rowsize - rowsize_scale)
    return fontsize

def plot_signal_ranges(ax, data, idx, fig_channels):
    if 'signal_ranges' in data.keys():
        merged_signal_ranges = merge_ranges(data['signal_ranges'][idx]) # to merge overlapping ranges
        for low, upp in merged_signal_ranges:
            ax.axvspan(fig_channels[low], fig_channels[upp - 1], alpha=0.1, color='indianred')
            
def get_title_string(idx, rchi2):
    rchi2_string = ''
    if rchi2 is not None:
        rchi2_string = ', $\\chi_{{red}}^{{2}}$={:.3f}'.format(rchi2[idx])
        
    title = 'Idx={}{}'.format(
        idx, rchi2_string)
    return title

def plot_spectra(fitsfiles, outfile='spectra.pdf', coordinates=None, radius=None, path_to_plots='.', n_spectra=9, rowsize=4., rowbreak=10, dpi=72, velocity_range=[-110,163], cunit3='m/s', vel_unit=u.km/u.s, seed=111):
    """
    fitsfiles: list of fitsfiles to plot spectra from
    coordinates: array of central coordinates [[Glon, Glat]] to plot spectra from
    radius: radius of area to be averaged for each spectrum [arcseconds]
    """
    
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
                    velocity = velocity_axes(fitsfile, cunit3=cunit3)
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
                    velocity = velocity_axes(fitsfile, cunit3=cunit3)
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
                if not isinstance(radius, str):
                    edge = int(np.ceil((radius/3600) / px_scale))
                else:
                    edge = 0.1 * min(xsize, ysize)
                xValue = rng.integers(edge+1,xsize-edge)
                yValue = rng.integers(edge+1,ysize-edge)
                ax = fig.add_subplot(rows,cols,i+1)
                for idx, fitsfile in enumerate(fitsfiles):
                    pixel_array = pixel_circle_calculation_px(fitsfile,x=xValue,y=yValue,r=radius)
                    spectrum = calculate_spectrum(fitsfile,pixel_array)
                    header = fits.getheader(fitsfile)
                    velocity = velocity_axes(fitsfile, cunit3=cunit3)
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
                    velocity = velocity_axes(fitsfile, cunit3=cunit3)
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
    """
    pickle_file: pickled file to plot spectra from
    """
    
    print("\nPlotting...")
    
    fontsize = scale_fontsize(rowsize)
    color_list, draw_list, line_list = styles_pickle()

    data = pickle_load_file(pickle_file)
    training_data = data['training_data']
    test_data = data['test_data']
    velocity = data['velocity']
    rms = data['rms_noise']
    if 'bg_fit' in data.keys():
        bg_fit = data['bg_fit']
    else:
        bg_fit = None
    if 'rchi2' in data.keys():
        rchi2 = data['rchi2']
    else:
        rchi2 = None
    if 'header' in data.keys():
        header = data['header']
    else:
        header = None
        
    rng = np.random.default_rng(seed)
    xsize = len(data['training_data'])
    cols, rows, rowbreak, colsize = get_figure_params(n_spectra, rowsize, rowbreak)
    figsize = (cols*colsize, rowbreak*rowsize)
    if bg_fit is not None:
        figsize = (cols*colsize, 1.5*rowbreak*rowsize)
    else:
        figsize = (cols*colsize, 1.2*rowbreak*rowsize)
    fig = plt.figure(figsize=figsize) #, constrained_layout=True
    gs0 = gridspec.GridSpec(rows, cols, figure=fig)
    xValue = rng.choice(xsize,size=n_spectra,replace=False)
    for i in trange(n_spectra):
        idx = xValue[i]
        velo_min, velo_max = find_nearest(velocity,np.amin(velocity_range)), find_nearest(velocity,np.amax(velocity_range))
        if bg_fit is not None:
            gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[i]) #, height_ratios=[3,1]
            ax = fig.add_subplot(gs00[0,0])
            ax.plot(velocity[velo_min:velo_max], test_data[idx][velo_min:velo_max], drawstyle=draw_list[0], color=color_list[0], linestyle=line_list[0], label="'pure' HI")
            ax.plot(velocity[velo_min:velo_max], training_data[idx][velo_min:velo_max], drawstyle=draw_list[1], color=color_list[1], linestyle=line_list[1], label="observed HI+HISA")
            ax.plot(velocity[velo_min:velo_max], bg_fit[idx][velo_min:velo_max], drawstyle=draw_list[2], color=color_list[2], linestyle=line_list[2], label="bg fit")
            add_figure_properties(ax, header=header, fontsize=fontsize, velocity_range=velocity_range, vel_unit=vel_unit, set_xlabel=False)
            #add residual plot
            ax2 = fig.add_subplot(gs00[1,0])
            ax2.plot(velocity[velo_min:velo_max], bg_fit[idx][velo_min:velo_max] - test_data[idx][velo_min:velo_max], drawstyle=draw_list[0], color=color_list[1], linestyle=line_list[0])
            ax2.set_title("Residual", fontsize=fontsize)
            ax2.axhline(color='black', ls='solid', lw=1.0)
            ax2.axhline(y=rms[idx], color='red', ls='dotted', lw=1.0)
            ax2.axhline(y=-rms[idx], color='red', ls='dotted', lw=1.0)
            plot_signal_ranges(ax2, data, idx, velocity)
            add_figure_properties(ax2, header=header, fontsize=fontsize, velocity_range=velocity_range, vel_unit=vel_unit, set_ylabel=False)
            #autoscale_y(fig)
            gs00.set_height_ratios([np.diff(ax.get_ylim())[0], np.diff(ax2.get_ylim())[0]])
        else:
            #ax = fig.add_subplot(rows,cols,i+1)
            ax = fig.add_subplot(gs0[i])
            ax.plot(velocity[velo_min:velo_max], test_data[idx][velo_min:velo_max], drawstyle=draw_list[0], color=color_list[0], linestyle=line_list[0], label="'pure' HI")
            ax.plot(velocity[velo_min:velo_max], training_data[idx][velo_min:velo_max], drawstyle=draw_list[1], color=color_list[1], linestyle=line_list[1], label="observed HI+HISA")
            add_figure_properties(ax, header=header, fontsize=fontsize, velocity_range=velocity_range, vel_unit=vel_unit)
        title = get_title_string(idx, rchi2)
        ax.set_title(title, fontsize=fontsize)
        plot_signal_ranges(ax, data, idx, velocity)
        ax.legend(loc=2, fontsize=fontsize-2)

    gs0.tight_layout(fig)
   # fig.tight_layout()

    if not os.path.exists(path_to_plots):
        os.makedirs(path_to_plots)
    if outfile is not None:
        filename = outfile
    elif outfile is None:
        #filename = pickle_file.split('/')[-1].split('.pickle')[0] + '_{}.pdf'.format(n_spectra)
        filename_wext = os.path.basename(pickle_file)
        filename_base, file_extension = os.path.splitext(filename_wext)
        filename = filename_base + '_{}.pdf'.format(n_spectra)
    pathname = os.path.join(path_to_plots, filename)
    fig.savefig(pathname, dpi=dpi, bbox_inches='tight')
    #plt.close()
    print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(filename, path_to_plots))
