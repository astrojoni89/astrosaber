# @Author: syed
# @Date:   2021-01
# @Filename: hisa.py
# @Last modified by:   syed
# @Last modified time: 10-06-2021

'''hisa extraction'''

import os
import sys
import numpy as np

from astropy.io import fits
from astropy import units as u

from tqdm import trange
import warnings

from .utils.aslsq_helper import count_ones_in_row, md_header_2d, check_signal_ranges, IterationWarning, say, format_warning
from .utils.aslsq_fit import baseline_als_optimized

warnings.showwarning = format_warning



class HisaExtraction(object):
    def __init__(self, fitsfile, path_to_noise_map=None, path_to_data='.', smoothing='Y', lam1=None, p1=None, lam2=None, p2=None, niters=20, iterations_for_convergence = 3, noise=None, add_residual = True, sig = 1.0, velo_range = 15.0, check_signal_sigma = 10, output_flags = True):
        self.fitsfile = fitsfile
        self.path_to_noise_map = path_to_noise_map
        self.path_to_data = path_to_data
        self.smoothing = smoothing
        
        self.lam1 = lam1
        self.p1 = p1
        self.lam2 = lam2
        self.p2 = p2
        
        self.niters = niters
        self.iterations_for_convergence = iterations_for_convergence
        
        self.noise = noise
        self.add_residual = add_residual
        self.sig = sig
        
        self.velo_range = velo_range
        self.check_signal_sigma = check_signal_sigma
        
        self.output_flags = output_flags

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
        string = 'Done!'
        say(string)

    #TODO
    def saber(self):
        self.prepare_data()

        if self.lam1 is None:
            raise TypeError("Need to specify 'lam1' for extraction.")
        if self.p1 is None:
            raise TypeError("Need to specify 'p1' for extraction.")
        if not 0<= self.p1 <=1:
            raise ValueError("'p1' has to be in the range [0,1]")
        if self.lam2 is None:
            raise TypeError("Need to specify 'lam2' for extraction.")
        if self.p2 is None:
            raise TypeError("Need to specify 'p2' for extraction.")
        if not 0<= self.p2 <=1:
            raise ValueError("'p2' has to be in the range [0,1]")

        if self.path_to_noise_map is not None:
            noise_map = fits.getdata(self.path_to_noise_map)
            thresh = self.sig * noise_map
        else:
            if self.noise is None:
               raise TypeError("Need to specify 'noise' if no path to noise map is given.") 
            else:
                noise_map = self.noise * np.ones((self.header['NAXIS2'],self.header['NAXIS1']))
                thresh = self.sig * noise_map

        pixel_start=[0,0]
        pixel_end=[self.header['NAXIS1'],self.header['NAXIS2']]

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
            for i in trange(pixel_start[0],pixel_end[0],1):
                for j in range(pixel_start[1],pixel_end[1],1):
                    spectrum = self.image[:,j,i]
                    if check_signal_ranges(spectrum, self.header, sigma=self.check_signal_sigma, noise=noise_map[j,i], velo_range=self.velo_range):
                        spectrum_prior = baseline_als_optimized(spectrum, self.lam1, self.p1, niter=3)
                        spectrum_firstfit = spectrum_prior
                        n = 0
                        converge_logic = np.array([])
                        while n < self.niters:
                            spectrum_prior = baseline_als_optimized(spectrum_prior, self.lam2, self.p2, niter=3)
                            spectrum_next = baseline_als_optimized(spectrum_prior, self.lam2, self.p2, niter=3)
                            residual = abs(spectrum_next - spectrum_prior)
                            if np.any(np.isnan(residual)):
                                print('Residual contains NaNs') 
                                residual[np.isnan(residual)] = 0.0
                            converge_test = (np.all(residual < thresh[j,i]))
                            converge_logic = np.append(converge_logic,converge_test)
                            c = count_ones_in_row(converge_logic)
                            if np.any(c > self.iterations_for_convergence):
                                i_converge = np.min(np.argwhere(c > self.iterations_for_convergence))
                                res = abs(spectrum_next - spectrum_firstfit)
                                if self.add_residual:
                                    final_spec = spectrum_next + res
                                else:
                                    final_spec = spectrum_next
                                break
                            else:
                                n += 1
                            if n==self.niters:
                                warnings.warn('Pixel (x,y)=({},{}). Maximum number of iterations reached. Fit did not converge.'.format(i,j), IterationWarning)
                                #flags
                                self.flag_map[j,i] = 0.
                                res = abs(spectrum_next - spectrum_firstfit)
                                if self.add_residual:
                                    final_spec = spectrum_next + res
                                else:
                                    final_spec = spectrum_next
                        self.image_asy[:,j,i] = final_spec - thresh[j,i]
                        self.HISA_map[:,j,i] = final_spec - self.image[:,j,i] - thresh[j,i]
                        self.iteration_map[j,i] = i_converge
                    else:
                        self.image_asy[:,j,i] = np.nan
                        self.HISA_map[:,j,i] = np.nan
                        self.iteration_map[j,i] = np.nan
                        #flags
                        self.flag_map[j,i] = 0.

            string = 'Done!'
            say(string)
            self.save_data()
            
        else:
            raise Exception("No smoothing applied. Set smoothing to 'Y'")
            
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
        
