import numpy as np
import pickle
from astropy.io import fits
from scipy import sparse
from scipy.sparse.linalg import spsolve

from tqdm import tqdm, trange
import warnings
import os

from .utils.quality_checks import goodness_of_fit, get_max_consecutive_channels, determine_peaks, mask_channels
from .utils.aslsq_helper import velocity_axes, count_ones_in_row, md_header_2d, check_signal_ranges, IterationWarning, say, format_warning
from .utils.aslsq_fit import baseline_als_optimized

from .parallel_processing import init, parallel_process, parallel_process_wo_bar, func, func_wo_bar

warnings.showwarning = format_warning

def gauss_function(x,amp,mu,sigma):
    return amp * np.exp(-(x-mu)**2/(2*sigma**2))



class saberPrepare(object):
    def __init__(self, fitsfile, training_set_size=100, path_to_noise_map=None, path_to_data='.', smoothing='Y', mean_linewidth=4.,std_linewidth=1., lam1=None, p1=None, lam2=None, p2=None, niters=20, iterations_for_convergence=3, noise=None, add_residual = True, sig = 1.0, velo_range = 15.0, check_signal_sigma = 6., output_flags = True, p_limit=None, ncpus=None, suffix='', filename_out=None):
        self.fitsfile = fitsfile
        self.training_set_size = int(training_set_size)
        self.path_to_noise_map = path_to_noise_map
        self.path_to_data = path_to_data
        self.smoothing = smoothing

        self.mean_linewidth = mean_linewidth
        self.std_linewidth = std_linewidth
        
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

        self.p_limit = p_limit

        self.ncpus = ncpus

        self.suffix = suffix
        self.filename_out = filename_out
        
    def __str__(self):
        return f'saberPrepare:\nfitsfile: {self.fitsfile}\ntraining_set_size: {self.training_set_size}\npath_to_noise_map: {self.path_to_noise_map}\npath_to_data: {self.path_to_data}\nsmoothing: {self.smoothing}\nlam1: {self.lam1}\np1: {self.p1}\nlam2: {self.lam2}\np2: {self.p2}\nniters: {self.niters}\niterations_for_convergence: {self.iterations_for_convergence}\nnoise: {self.noise}\nadd_residual: {self.add_residual}\nsig: {self.sig}\nvelo_range: {self.velo_range}\ncheck_signal_sigma: {self.check_signal_sigma}\noutput_flags: {self.output_flags}\np_limit: {self.p_limit}\nncpus: {self.ncpus}'
    
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
        #self.header_2d = md_header_2d(self.fitsfile)
        self.v = self.header['NAXIS3']
        self.velocity = velocity_axes(self.fitsfile)
        self.mock_data = {'training_data' : None, 'test_data' : None, 'hisa_spectra' : None, 'velocity' : None, 'header' : None}
        self.hisa_spectra = []
        self.training_data = []
        self.test_data = []
        string = 'Done!'
        say(string)

    def prepare_training(self):
        np.random.seed(111)
        self.prepare_data()

        if self.training_set_size <= 0:
            raise ValueError("'training_set_size' has to be >0")
        if self.lam1 is None:
            self.lam1 = 0.50
        if self.p1 is None:
            self.p1 = 0.90
        if not 0<= self.p1 <=1:
            raise ValueError("'p1' has to be in the range [0,1]")
        if self.lam2 is None:
            self.lam2 = 1.00
        if self.p2 is None:
            self.p2 = 0.90
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

        if self.p_limit is None:
            self.p_limit = 0.02

        self.max_consec_ch = get_max_consecutive_channels(self.v, self.p_limit)
        channel_width = self.header['CDELT3'] / 1000.
        spectral_resolution = 1/np.sqrt(8*np.log(2)) # sigma; unit channel
        edges = int(0.10 * min(self.header['NAXIS1'],self.header['NAXIS2']))
        indices = np.column_stack((np.random.randint(edges,self.header['NAXIS2']-edges,self.training_set_size), np.random.randint(edges,self.header['NAXIS1']-edges,self.training_set_size)))

        mu_lws_HISA, sigma_lws_HISA = self.mean_linewidth/np.sqrt(8*np.log(2)) / channel_width, self.std_linewidth/np.sqrt(8*np.log(2)) / channel_width # mean and standard deviation
        mu_ncomps_HISA, sigma_ncomps_HISA = 2, 1 
        lws_HISA = np.random.normal(mu_lws_HISA, sigma_lws_HISA, self.training_set_size).reshape(self.training_set_size,)
        ncomps_HISA = np.random.normal(mu_ncomps_HISA, sigma_ncomps_HISA, self.training_set_size).reshape(self.training_set_size).astype(int)
        ncomps_HISA[ncomps_HISA<0] = int(1.)

        xvals = np.arange(0,self.v,1)
        
        self.spectrum_list = []
        self.noise_list = []
        self.thresh_list = []
        for idx, (y, x) in enumerate(zip(indices[:,0], indices[:,1])):
            self.spectrum_list.append(self.image[:,y,x])
            self.noise_list.append(noise_map[y,x])
            self.thresh_list.append(thresh[y,x])
        init(self.spectrum_list)

        self.results_list = func(use_ncpus=self.ncpus, function=self.two_step_extraction) # initiate parallel process

        for i in trange(len(self.results_list[:,0])):
            amps_HISA = np.random.normal(results_list[i,3], results_list[i,4], self.training_set_size).reshape(self.training_set_size,)
            amps_HISA[amps_HISA<0] = 0.
            mu_velos_HISA, sigma_velos_HISA = min(results_list[i,1][:,0]) + np.abs(max(results_list[i,1][:,1]) - min(results_list[i,1][:,0])) / 2., 5. # mean and standard deviation
            velos_HISA = np.random.normal(mu_velos_HISA, sigma_velos_HISA, self.training_set_size).reshape(self.training_set_size,)
            velos_of_comps_HISA = np.random.choice(velos_HISA, ncomps_HISA[i])
            amps_of_comps_HISA = np.random.choice(amps_HISA, ncomps_HISA[i])
            lws_of_comps_HISA = np.random.choice(lws_HISA, ncomps_HISA[i])  
            ncomp_HISA = np.arange(0,ncomps_HISA[i],1)
            lws_of_comps_HISA[np.argwhere(lws_of_comps_HISA<spectral_resolution)] = spectral_resolution

            gauss_HISA = np.zeros(shape=(self.v,))
            for idx, (c, v, lw, amp) in enumerate(zip(ncomp_HISA,velos_of_comps_HISA,lws_of_comps_HISA,amps_of_comps_HISA)):
                gauss_HISA = gauss_HISA + gauss_function(xvals,amp, v, lw)

            #limit HISA to HI emission
            for ch in range(len(gauss_HISA)):
                if gauss_HISA[ch]>mock_emission[ch]:
                    gauss_HISA[ch] = mock_emission[ch]
            gauss_HISA[np.invert(mask)] = 0.   

        training_data.append(mock_emission - gauss_HISA)
        test_data.append(mock_emission)
        hisa_spectra.append(gauss_HISA)

        self.mock_data['training_data'] = training_data
        self.mock_data['test_data'] = test_data
        self.mock_data['hisa_spectra'] = hisa_spectra
        self.mock_data['velocity'] = self.velocity
        self.mock_data['header'] = self.header

        self.save_data()


    def two_step_extraction(self, i):
        flag = 1.
        if check_signal_ranges(self.spectrum_list[i], self.header, sigma=self.check_signal_sigma, noise=self.noise_list[i], velo_range=self.velo_range):
            spectrum_prior = baseline_als_optimized(self.spectrum_list[i], self.lam1, self.p1, niter=3)
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
                converge_test = (np.all(residual < self.thresh_list[i]))
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
                    warnings.warn('Maximum number of iterations reached. Fit did not converge.', IterationWarning)
                    #flags
                    flag = 0.
                    res = abs(spectrum_next - spectrum_firstfit)
                    if self.add_residual:
                        final_spec = spectrum_next + res
                    else:
                        final_spec = spectrum_next
                    i_converge = self.niters
            bg = final_spec - self.thresh_list[i]
            hisa = final_spec - self.spectrum_list[i] - self.thresh_list[i]
            iterations = i_converge
        else:
            bg = np.nan
            hisa = np.nan
            iterations = np.nan
            #flags
            flag = 0.

        consecutive_channels, ranges = determine_peaks(spectrum=self.spectrum_list[i], peak='positive', amp_threshold=None)
        #amp_values, ranges = determine_peaks(spectrum=data[:,y,x], peak='positive', amp_threshold=6*self.noise_list[i])
        mask_ranges = ranges[np.where(consecutive_channels>=self.max_consec_ch)]
        mask = mask_channels(self.v, mask_ranges, pad_channels=0, remove_intervals=None)
        
        obs_noise = np.random.normal(0,self.noise_list[i],size=(self.v,))
        mock_emission = bg + obs_noise

        mu_amps_HISA, sigma_amps_HISA = 6*self.noise_list[i], 1*self.noise_list[i]

        return mock_emission, mask_ranges, mask, mu_amps_HISA, sigma_amps_HISA


    def save_data(self):
        if filename_out is None:
            filename_out = '{}-training_set-{}_spectra{}.pickle'.format(self.fitsfile.split('/')[-1].split('.fits')[0], self.training_set_size, self.suffix)
        elif not filename_out.endswith('.pickle'):
            filename_out = filename_out + '.pickle'
        dirname = os.path.join(self.path_to_data, 'astrosaber_training')
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        path_to_file = os.path.join(dirname, filename_out)
        pickle.dump(mock_data, open(path_to_file, 'wb'), protocol=2)
        say("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(filename_out, dirname))

