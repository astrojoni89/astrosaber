# @Author: syed
# @Date:   2021-01
# @Filename: hisa.py
# @Last modified by:   syed
# @Last modified time: 05-02-2022

'''hisa extraction'''

import os
import sys
import numpy as np
from pathlib import Path
from typing import Type, Optional, List, Tuple

from astropy.io import fits
from astropy import units as u

from tqdm import tqdm
from tqdm.utils import _is_utf, _supports_unicode
import warnings

from .utils.aslsq_helper import count_ones_in_row, md_header_2d, check_signal_ranges, IterationWarning, say, format_warning
from .utils.aslsq_fit import baseline_als_optimized, two_step_extraction, one_step_extraction
from .utils.grogu import yoda

warnings.showwarning = format_warning



class HisaExtraction(object):
    """
    A class used to execute the self-absorption extraction

    ...

    Attributes
    ----------
    fitsfile : str
        Name of the fitsfile.
    path_to_noise_map : Path
        Path to the noise map.
    path_to_data : Path
        Path to the fitsfile.
    smoothing : bool, optional
        Whether to execute the asymmetric least squares smoothing routine.
    phase : str, optional
        Mode of saber smoothing, eihter 'one' or 'two' (default) phase smoothing.
    lam1 : float
        Lambda_1 smoothing parameter.
    p1 : float
        Asymmetry weight of the minor cycle smoothing.
    lam2 : float
        Lambda_2 smoothing parameter. Has to be specified if phase is set to 'two'.
    p2 : float
        Asymmetry weight of the major cycle smoothing. Has to be specified if phase is set to 'two'.
    niters : int, optional
        Maximum number of iterations of the smoothing.
    iterations_for_convergence : int, optional
        Number of iterations of the major cycle for the baseline to be considered converged.
    noise : float
        Noise level of the data. Has to be specified if no path to noise map is given.
    add_residual : bool, optional
        Whether to add the residual (=difference between first and last major cycle iteration) to the baseline. Default is True.
    sig : float, optional
        Defines how many sigma of the noise is used as a convergence criterion.
        If change in baseline between major cycle iterations is smaller than 'sig' * 'noise' for 'iterations_for_convergence',
        then the baseline is considered converged.
    velo_range : float, optional
        Velocity range [in km/s] of the spectra that has to contain significant signal
        for it to be considered in the baseline extraction. Default is 15.0.
    check_signal_sigma : float, optional
        Defines the significance of the signal that has to be present in the spectra
        for at least the range defined by 'velo_range'. Default is 6.0.
    output_flags : bool, optional
        Whether to save a mask containing the flags. Default is True.
    baby_yoda : bool, optional
        Whether to show a star wars-themed progress bar. Default is False.
    p_limit : float
        The p-limit of the Markov chain to estimate signal ranges in the spectra.
    ncpus : int
        Number of CPUs to use.
    suffix : str, optional
        Optional suffix to add to the output filenames.

    Methods
    -------
    #TODO
    """
    def __init__(self, fitsfile : str, path_to_noise_map : Path = None, path_to_data : Path = '.',
                 smoothing : Optional[str] = 'Y', phase : Optional[str] = 'two', lam1 : float = None, p1 : float = None,
                 lam2 : float = None, p2 : float = None, niters : Optional[int] = 20,
                 iterations_for_convergence : Optional[int] = 3, noise : float = None,
                 add_residual : Optional[bool] = True, sig : Optional[float] = 1.0, velo_range : Optional[float] = 15.0,
                 check_signal_sigma : Optional[float] = 6., output_flags : Optional[bool] = True, baby_yoda : Optional[bool] = False,
                 p_limit : float = None, ncpus : int = None, suffix : Optional[str] = ''):
        
        self.fitsfile = fitsfile
        self.path_to_noise_map = path_to_noise_map
        self.path_to_data = path_to_data
        self.smoothing = smoothing
        self.phase = phase
        
        self.lam1 = lam1
        self.p1 = p1
        self.lam2 = lam2
        self.p2 = p2
        
        self.niters = int(niters)
        self.iterations_for_convergence = int(iterations_for_convergence)
        
        self.noise = noise
        self.add_residual = add_residual
        self.sig = sig
        
        self.velo_range = velo_range
        self.check_signal_sigma = check_signal_sigma
        
        self.output_flags = output_flags
        
        self.baby_yoda = baby_yoda #NO IDEA WHAT THIS DOES
        
        self.p_limit = p_limit
        
        self.ncpus = ncpus
        
        self.suffix = suffix
        
    def __repr__(self):
        return f'HisaExtraction(
                 \nfitsfile: {self.fitsfile}
                 \npath_to_noise_map: {self.path_to_noise_map}
                 \npath_to_data: {self.path_to_data}
                 \nsmoothing: {self.smoothing}
                 \nphase: {self.phase}
                 \nlam1: {self.lam1}
                 \np1: {self.p1}
                 \nlam2: {self.lam2}
                 \np2: {self.p2}
                 \nniters: {self.niters}
                 \niterations_for_convergence: {self.iterations_for_convergence}
                 \nnoise: {self.noise}
                 \nadd_residual: {self.add_residual}
                 \nsig: {self.sig}
                 \nvelo_range: {self.velo_range}
                 \ncheck_signal_sigma: {self.check_signal_sigma}
                 \noutput_flags: {self.output_flags}
                 \np_limit: {self.p_limit}
                 \nncpus: {self.ncpus})'

    def getting_ready(self):
        string = 'preparation'
        banner = len(string) * '='
        heading = '\n' + banner + '\n' + string + '\n' + banner
        say(heading)

    def prepare_data(self):
        self.getting_ready()
        self.image = fits.getdata(self.fitsfile) #load data
        self.image[np.where(np.isnan(self.image))] = 0.0

        self.header = fits.getheader(self.fitsfile)
        self.header_2d = md_header_2d(self.fitsfile)
        self.v = self.header['NAXIS3']
        
        #serialize data to have a list of spectra
        self.list_data = []
        for y in range(self.image.shape[1]): 
            for x in range(self.image.shape[2]):
                idx_1d = np.ravel_multi_index((y,x), dims=(self.image.shape[1], self.image.shape[2]))
                self.list_data.append([idx_1d, self.image[:,y,x]])
        
        string = 'Done!'
        say(string)

    #TODO
    def saber(self):
        self.prepare_data()

        if self.lam1 is None:
            raise TypeError("Need to specify 'lam1' for extraction.")
        if self.p1 is None:
            self.p1 = 0.90
        if not 0<= self.p1 <=1:
            raise ValueError("'p1' has to be in the range [0,1]")
        if self.lam2 is None and self.phase=='two':
            raise TypeError("Need to specify 'lam2' for two-phase extraction.")
        if self.p2 is None:
            self.p2 = 0.90
        if not 0<= self.p2 <=1:
            raise ValueError("'p2' has to be in the range [0,1]")
        if self.p_limit is None:
            self.p_limit = 0.02

        if self.path_to_noise_map is not None:
            noise_map = fits.getdata(self.path_to_noise_map)
            thresh = self.sig * noise_map
            self.list_data_noise = []
            self.list_data_thresh = []
            for y in range(noise_map.shape[0]):
                for x in range(noise_map.shape[1]):
                    idx_1d = np.ravel_multi_index((y,x), dims=(noise_map.shape[0], noise_map.shape[1]))
                    self.list_data_noise.append([idx_1d, noise_map[y,x]])
                    self.list_data_thresh.append([idx_1d, thresh[y,x]])
        else:
            if self.noise is None:
               raise TypeError("Need to specify 'noise' if no path to noise map is given.") 
            else:
                noise_map = self.noise * np.ones((self.header['NAXIS2'],self.header['NAXIS1']))
                thresh = self.sig * noise_map
                self.list_data_noise = []
                self.list_data_thresh = []
                for y in range(noise_map.shape[0]):
                    for x in range(noise_map.shape[1]):
                        idx_1d = np.ravel_multi_index((y,x), dims=(noise_map.shape[0], noise_map.shape[1]))
                        self.list_data_noise.append([idx_1d, noise_map[y,x]])
                        self.list_data_thresh.append([idx_1d, thresh[y,x]])
                
        if self.baby_yoda:
            if _supports_unicode(sys.stderr):
                fran = yoda
            else:
                fran = tqdm
        else:
            fran = tqdm

        if self.smoothing=='Y':
            string = 'hisa extraction'
            banner = len(string) * '='
            heading = '\n' + banner + '\n' + string + '\n' + banner
            say(heading)

            self.image_asy = np.zeros((self.v,self.header['NAXIS2'],self.header['NAXIS1']))
            self.HISA_map = np.zeros((self.v,self.header['NAXIS2'],self.header['NAXIS1']))
            self.iteration_map = np.zeros((self.header['NAXIS2'],self.header['NAXIS1']))
            #flags
            self.flag_map = np.ones((self.header['NAXIS2'],self.header['NAXIS1']))
            
            print('\n'+'Asymmetric least squares fitting in progress...')
            
            if self.phase == 'two':
                import astroSABER.parallel_processing
                astroSABER.parallel_processing.init([self.list_data, [self]])
                results_list = astroSABER.parallel_processing.func(use_ncpus=self.ncpus, function='two_step', bar=fran) 
                    
            elif self.phase == 'one':
                import astroSABER.parallel_processing
                astroSABER.parallel_processing.init([self.list_data, [self]])
                results_list = astroSABER.parallel_processing.func(use_ncpus=self.ncpus, function='one_step', bar=fran)
                
            print('\n'+'Unraveling data and writing into cubes...')
            for k in tqdm(range(len(results_list)), unit='spec', unit_scale=True):
                (j, i) = np.unravel_index(results_list[k][0], (self.HISA_map.shape[1], self.HISA_map.shape[2]))   
                self.image_asy[:,j,i], self.HISA_map[:,j,i], self.iteration_map[j,i], self.flag_map[j,i] = results_list[k][1], results_list[k][2], results_list[k][3], results_list[k][4]
            
            string = 'Done!'
            say(string)
            self.save_data()
            
        else:
            raise Exception("No smoothing applied. Set smoothing to 'Y'")
        
        
    def two_step_extraction_single(self, i):
        bg, hisa, iterations, flag_map = two_step_extraction(self.lam1, self.p1, self.lam2, self.p2, spectrum=self.list_data[i][1], header=self.header, check_signal_sigma=self.check_signal_sigma, noise=self.list_data_noise[i][1], velo_range=self.velo_range, niters=self.niters, iterations_for_convergence=self.iterations_for_convergence, add_residual=self.add_residual, thresh=self.list_data_thresh[i][1], p_limit=self.p_limit)
        return self.list_data[i][0], bg, hisa, iterations, flag_map
    
    
    def one_step_extraction_single(self, i):
        bg, hisa, iterations, flag_map = one_step_extraction(self.lam1, self.p1, spectrum=self.list_data[i][1], header=self.header, check_signal_sigma=self.check_signal_sigma, noise=self.list_data_noise[i][1], velo_range=self.velo_range, niters=self.niters, iterations_for_convergence=self.iterations_for_convergence, add_residual=self.add_residual, thresh=self.list_data_thresh[i][1], p_limit=self.p_limit)
        return self.list_data[i][0], bg, hisa, iterations, flag_map
       
        
    def save_data(self):
        filename_wext = os.path.basename(self.fitsfile)
        filename_base, file_extension = os.path.splitext(filename_wext)
        filename_bg = filename_base + '_aslsq_bg_spectrum{}.fits'.format(self.suffix)
        filename_hisa = filename_base + '_HISA_spectrum{}.fits'.format(self.suffix)
        filename_iter = filename_base + '_number_of_iterations{}.fits'.format(self.suffix)
        #flags
        filename_flags = filename_base + '_flags{}.fits'.format(self.suffix)
        
        if not os.path.exists(self.path_to_data):
            os.makedirs(self.path_to_data)
        pathname_bg = os.path.join(self.path_to_data, filename_bg)
        fits.writeto(pathname_bg, self.image_asy, header=self.header, overwrite=True)
        print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(filename_bg, self.path_to_data))
        pathname_hisa = os.path.join(self.path_to_data, filename_hisa)
        fits.writeto(pathname_hisa, self.HISA_map, header=self.header, overwrite=True)
        print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(filename_hisa, self.path_to_data))
        pathname_iter = os.path.join(self.path_to_data, filename_iter)
        fits.writeto(pathname_iter, self.iteration_map, header=self.header_2d, overwrite=True)
        print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(filename_iter, self.path_to_data))
        #flags
        pathname_flags = os.path.join(self.path_to_data, filename_flags)
        if self.output_flags:
            fits.writeto(pathname_flags, self.flag_map, header=self.header_2d, overwrite=True)
            print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(filename_flags, self.path_to_data))
        
