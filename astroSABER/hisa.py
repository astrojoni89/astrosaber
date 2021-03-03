# @Author: syed
# @Date:   2021-01
# @Filename: hisa.py
# @Last modified by:   syed
# @Last modified time: 15-02-2021

'''hisa extraction'''

import os
import numpy as np

from astropy.io import fits

from tqdm import trange
import warnings

from .utils.aslsq_helper import count_ones_in_row, md_header_2d, check_signal_ranges, IterationWarning, say

from .utils.aslsq_fit import baseline_als_optimized


class HisaExtraction(object):
    def __init__(self, fitsfile, path_to_noise_map=None):
        self.fitsfile = fitsfile
        self.path_to_noise_map = path_to_noise_map
        self.path_to_data = '.'
        self.smoothing = 'Y'
        self.lam1 = None
        self.p1 = None
        self.lam2 = None
        self.p2 = None
        self.niters = 20
        self.iterations_for_convergence = 3
        self.noise = None

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
            raise Exception("Need to specify 'lam1' for extraction.")
        if self.p1 is None:
            raise Exception("Need to specify 'p1' for extraction.")
        if self.lam2 is None:
            raise Exception("Need to specify 'lam2' for extraction.")
        if self.p2 is None:
            raise Exception("Need to specify 'p2' for extraction.")

        if self.path_to_noise_map is not None:
            noise_map = fits.getdata(self.path_to_noise_map)
            thresh = 1.0 * noise_map
        else:
            if self.noise is None:
               raise Exception("Need to specify 'noise' if no path to noise map is given.") 
            else:
                noise_map = self.noise * np.ones((self.header['NAXIS2'],self.header['NAXIS1']))
                thresh = 1.0 * noise_map

        pixel_start=[0,0]
        pixel_end=[self.header['NAXIS1'],self.header['NAXIS2']]

        if self.smoothing=='Y':
            string = 'hisa extraction'
            banner = len(string) * '='
            heading = '\n' + banner + '\n' + string + '\n' + banner
            say(heading)

            image_asy = np.zeros((self.v,self.header['NAXIS2'],self.header['NAXIS1']))
            HISA_map = np.zeros((self.v,self.header['NAXIS2'],self.header['NAXIS1']))
            iteration_map = np.zeros((self.header['NAXIS2'],self.header['NAXIS1']))
            print('\n'+'Asymmetric least squares fitting in progress...')
            for i in trange(pixel_start[0],pixel_end[0],1):
                for j in range(pixel_start[1],pixel_end[1],1):
                    spectrum = self.image[:,j,i]
                    if check_signal_ranges(spectrum, self.header, sigma=10, noise=noise_map[j,i], velo_range=15.0):
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
                                final_spec = spectrum_next + res
                                break
                            else:
                                n += 1
                            if n==self.niters:
                                warnings.warn('Pixel (x,y)=({},{}). Maximum number of iterations reached. Fit did not converge.'.format(i,j), IterationWarning)
                                res = abs(spectrum_next - spectrum_firstfit)
                                final_spec = spectrum_next + res
                        image_asy[:,j,i] = final_spec - thresh[j,i]
                        HISA_map[:,j,i] = final_spec - self.image[:,j,i] - thresh[j,i]
                        iteration_map[j,i] = i_converge
                    else:
                        image_asy[:,j,i] = np.nan
                        HISA_map[:,j,i] = np.nan
                        iteration_map[j,i] = np.nan

            stri = 'Done!'
            say(stri)
            filename_bg = self.fitsfile.split('.fits')[0]+'_aslsq_bg_spectrum.fits'
            filename_hisa = self.fitsfile.split('.fits')[0]+'_HISA_spectrum.fits'
            filename_iter = self.fitsfile.split('.fits')[0]+'_number_of_iterations.fits'
            pathname_bg = os.path.join(self.path_to_data, filename_bg)
            fits.writeto(pathname_bg, image_asy, header=self.header, overwrite=True)
            print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(filename_bg, self.path_to_data))
            pathname_hisa = os.path.join(self.path_to_data, filename_hisa)
            fits.writeto(pathname_hisa, HISA_map, header=self.header, overwrite=True)
            print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(filename_hisa, self.path_to_data))
            pathname_iter = os.path.join(self.path_to_data, filename_iter)
            fits.writeto(pathname_iter, iteration_map, header=self.header_2d, overwrite=True)
            print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(filename_iter, self.path_to_data))
        else:
            raise Exception("No smoothing applied. Set smoothing to 'Y'")

