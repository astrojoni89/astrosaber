# @Author: syed
# @Date:   2021-01
# @Filename: hisa.py
# @Last modified by:   syed
# @Last modified time: 05-02-2022

'''hisa extraction'''

import os
import sys
import numpy as np

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
    def __init__(self, fitsfile, path_to_noise_map=None, path_to_data='.', smoothing='Y', phase='two', lam1=None, p1=None, lam2=None, p2=None, niters=50, iterations_for_convergence = 3, noise=None, add_residual = True, sig = 1.0, velo_range = 15.0, check_signal_sigma = 6., output_flags = True, baby_yoda = False, ncpus=None):
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
        
        self.ncpus = ncpus
        
    def __str__(self):
        return f'HisaExtraction:\nfitsfile: {self.fitsfile}\npath_to_noise_map: {self.path_to_noise_map}\npath_to_data: {self.path_to_data}\nsmoothing: {self.smoothing}\nphase: {self.phase}\nlam1: {self.lam1}\np1: {self.p1}\nlam2: {self.lam2}\np2: {self.p2}\nniters: {self.niters}\niterations_for_convergence: {self.iterations_for_convergence}\nnoise: {self.noise}\nadd_residual: {self.add_residual}\nsig: {self.sig}\nvelo_range: {self.velo_range}\ncheck_signal_sigma: {self.check_signal_sigma}\noutput_flags: {self.output_flags}\nncpus: {self.ncpus}'

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
        if self.lam2 is None:
            raise TypeError("Need to specify 'lam2' for extraction.")
        if self.p2 is None:
            self.p2 = 0.90
        if not 0<= self.p2 <=1:
            raise ValueError("'p2' has to be in the range [0,1]")

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
                results_list = astroSABER.parallel_processing.func(use_ncpus=self.ncpus, function='two_step') 
                    
            elif self.phase == 'one':
                import astroSABER.parallel_processing
                astroSABER.parallel_processing.init([self.list_data, [self]])
                results_list = astroSABER.parallel_processing.func(use_ncpus=self.ncpus, function='one_step')
                
            print('\n'+'Unraveling data and writing into cubes...'+'\n')
            for k in fran(range(len(results_list))):
                (j, i) = np.unravel_index(results_list[k][0], (self.HISA_map.shape[1], self.HISA_map.shape[2]))   
                self.image_asy[:,j,i], self.HISA_map[:,j,i], self.iteration_map[j,i], self.flag_map[j,i] = results_list[k][1], results_list[k][2], results_list[k][3], results_list[k][4]
            
            string = 'Done!'
            say(string)
            self.save_data()
            
        else:
            raise Exception("No smoothing applied. Set smoothing to 'Y'")
        
        
    def two_step_extraction_single(self, i):
        bg, hisa, iterations, flag_map = two_step_extraction(self.lam1, self.p1, self.lam2, self.p2, spectrum=self.list_data[i][1], header=self.header, check_signal_sigma=self.check_signal_sigma, noise=self.list_data_noise[i][1], velo_range=self.velo_range, niters=self.niters, iterations_for_convergence=self.iterations_for_convergence, add_residual=self.add_residual, thresh=self.list_data_thresh[i][1])
        return self.list_data[i][0], bg, hisa, iterations, flag_map
    
    
    def one_step_extraction_single(self, i):
        bg, hisa, iterations, flag_map = one_step_extraction(self.lam1, self.p1, spectrum=self.list_data[i][1], header=self.header, check_signal_sigma=self.check_signal_sigma, noise=self.list_data_noise[i][1], velo_range=self.velo_range, niters=self.niters, iterations_for_convergence=self.iterations_for_convergence, add_residual=self.add_residual, thresh=self.list_data_thresh[i][1])
        return self.list_data[i][0], bg, hisa, iterations, flag_map
       
        
    def save_data(self):
        filename_bg = self.fitsfile.split('/')[-1].split('.fits')[0]+'_aslsq_bg_spectrum.fits'
        filename_hisa = self.fitsfile.split('/')[-1].split('.fits')[0]+'_HISA_spectrum.fits'
        filename_iter = self.fitsfile.split('/')[-1].split('.fits')[0]+'_number_of_iterations.fits'
        #flags
        filename_flags = self.fitsfile.split('/')[-1].split('.fits')[0]+'_flags.fits'
        
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
        
