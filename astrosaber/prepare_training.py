import numpy as np
import pickle
from pathlib import Path
from typing import Union, Optional, List, Tuple

from astropy.io import fits
from astropy import units as u

from tqdm import trange
import warnings
import os

from .utils.quality_checks import get_max_consecutive_channels, determine_peaks, mask_channels
from .utils.aslsq_helper import find_nearest, velocity_axes, say, format_warning
from .utils.aslsq_fit import two_step_extraction
from .plotting import plot_pickle_spectra

warnings.showwarning = format_warning
np.seterr('raise')



class saberPrepare(object):
    """
    A class used to obtain and prepare training data for the optimization.
    
    Attributes
    ----------
    fitsfile : str
        Name of the fitsfile.
    training_set_size : int, optional
        Number of spectra to draw from the data.
        Default is 100.
    path_to_noise_map : Path
        Path to the noise map.
        If no noise map is given, a single value must be provided instead using the 'noise' attribute.
    path_to_data : Path
        Path to the fitsfile.
    mean_amp_snr : int | float, optional
        Mean of amplitude distribution.
        This sets the mean value of the Gaussian distribution of self-absorption amplitude values
        from which the training data are generated. Default is 7.0.
    std_amp_snr : int | float, optional
        Standard deviation of amplitude distribution.
        This sets the standard deviation of the Gaussian distribution of self-absorption amplitude values
        from which the training data are generated. Default is 1.0.
    mean_linewidth : int | float, optional
        Mean of linewidth distribution in units of km/s [FWHM].
        This sets the mean value of the Gaussian distribution of self-absorption linewidths
        from which the training data are generated. Default is None.
    std_linewidth : int | float, optional
        Standard deviation of linewidth distribution in units of km/s [FWHM].
        This sets the standard deviation of the Gaussian distribution of self-absorption linewidths
        from which the training data are generated. Default is None.
    mean_ncomponent : int | float, optional
        Mean number of self-absorption components.
        This sets the mean value of the Gaussian distribution
        from which the number of self-absorption features added per spectrum is determined.
        Default is 2.0.
    std_ncomponent : int | float, optional
        Standard deviation of self-absorption components.
        This sets the standard deviation of the Gaussian distribution
        from which the number of self-absorption features added per spectrum is determined.
        Default is 0.5.
    fix_velocities : list, optional
        List of fixed velocities where self-absorption features should be added (in units of the third axis in the fits file).
        If the velocities of absorption features are known, this information can be provided here to aid the optimization.
    fix_velocities_sigma : float, optional
        Standard deviation of the fixed velocities.
        If a list of fixed velocities is given, some 'wiggle room' defined by this attribute can be added.
        This is the standard deviation of a Gaussian distribution around the fixed velocities
        in units of the third axis of the fits file. The default is one spectral channel.
    smooth_testdata: bool, optional
        Option to apply prior smoothing to data when generating test data.
        If `True`, prior smoothing will be applied using the `saberPrepare.lam1` and `saberPrepare.lam2` attributes.
        If `False`, original data will be used as test data. Default is `True`.
    lam1 : float, optional
        Lambda_1 smoothing parameter to generate test data.
        Default is 2.0.
    p1 : float, optional
        Asymmetry weight of the first iteration of the major cycle smoothing to generate test data.
        Default is 0.90.
    lam2 : float, optional
        Lambda_2 smoothing parameter to generate test data.
        Default is 2.0.
    p2 : float, optional
        Asymmetry weight of the remaining iterations of the major cycle smoothing to generate test data.
        Default is 0.90.
    niters : int, optional
        Maximum number of iterations of the smoothing.
        Only used to generate test data. Default is 20.
    iterations_for_convergence : int, optional
        Number of iterations of the major cycle for the baseline to be considered converged.
        Only used to generate test data. Default is 3.
    noise : float
        Noise level of the data. Has to be specified if no path to noise map is given.
    add_residual : bool, optional
        Whether to add the residual (=difference between first and last major cycle iteration) to the baseline.
        Only used to generate test data. Default is `True`.
    sig : float, optional
        Defines how many sigma of the noise is used as a convergence criterion.
        If the change in baseline between major cycle iterations is smaller than `sig` * `noise` for `iterations_for_convergence`,
        then the baseline is considered converged. Only used to generate test data. Default is 1.0.
    velo_range : float, optional
        Velocity range [in km/s] of the spectra that has to contain significant signal
        for it to be considered in the baseline extraction. Default is 15.0.
    check_signal_sigma : float, optional
        Defines the significance of the signal that has to be present in the spectra
        for at least the range defined by `velo_range`. Default is 6.0.
    p_limit : float, optional
        The p-limit of the Markov chain to estimate signal ranges in the spectra.
        Default is 0.01.
    ncpus : int, optional
        Number of CPUs to use.
        Defaults to 1.
    suffix : str, optional
        Optional suffix to add to the output filenames.
    filename_out : str, optional
        Output filename of the pickled file that contains the training and test data.
        The default is the fits filename base with the number of training spectra.
    path_to_file : str, optional
        Optional path to where the pickled training data should be stored.
        The training data are stored by default in the subfolder 'astrosaber_training'
        of the working directory.
    seed : int, optional
        Seed to initialize the random generator.

    Methods
    -------
    getting_ready()
        Prints a message when preparation starts.
    prepare_data()
        Prepares the training data by reading in data and
        setting up lists for Gaussian parameter distributions.
    prepare_training()
        Creates the test and training data and saves them into a pickled file.
    two_step_extraction_prepare()
        Runs the two-phase smoothing with default parameters to generate test data and self-absorption parameters.
    save_data()
        Saves all the test and training data into a pickled file.
    """

    def __init__(self, fitsfile : str, training_set_size : Union[int, float] = 100,
                 path_to_noise_map : Path =None, path_to_data : Path = '.',
                 mean_amp_snr : Union[int, float] = 7., std_amp_snr : Union[int, float] = 1.,
                 mean_linewidth : Union[int, float] = None, std_linewidth : Union[int, float] = None,
                 mean_ncomponent : Union[int, float] = 2., std_ncomponent : Union[int, float] = .5,
                 fix_velocities : Optional[List] = None, fix_velocities_sigma : Optional[float] = None,
                 smooth_testdata : bool = True,
                 lam1 : Optional[float] = None, p1 : Optional[float] = None,
                 lam2 : Optional[float] = None, p2 : Optional[float] = None,
                 niters : Optional[int] = 20, iterations_for_convergence : Optional[int] = 3,
                 noise : float = None, add_residual : Optional[bool] = False, sig : Optional[float] = 1.0,
                 velo_range : Optional[float] = 15.0, check_signal_sigma : Optional[float] = 6.,
                 p_limit : Optional[float] = 0.01, ncpus : Optional[int] = 1,
                 suffix : Optional[str] = '', filename_out : Optional[str] = None,
                 path_to_file : Optional[Path] = '.', seed : Optional[int] = 111):
        
        self.fitsfile = fitsfile
        self.training_set_size = int(training_set_size)
        self.path_to_noise_map = path_to_noise_map
        self.path_to_data = path_to_data

        self.mean_amp_snr = mean_amp_snr
        self.std_amp_snr = std_amp_snr
        self.mean_linewidth = mean_linewidth
        self.std_linewidth = std_linewidth
        self.mean_ncomponent = mean_ncomponent
        self.std_ncomponent = std_ncomponent
        
        self.fix_velocities = fix_velocities
        self.fix_velocities_sigma = fix_velocities_sigma
        
        self.smooth_testdata = smooth_testdata
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

        self.p_limit = p_limit

        self.ncpus = ncpus

        self.suffix = suffix
        self.filename_out = filename_out
        self.path_to_file = path_to_file
        
        self.seed = seed
        
    def __repr__(self):
        return f'''saberPrepare(
                fitsfile: {self.fitsfile}
                training_set_size: {self.training_set_size}
                path_to_noise_map: {self.path_to_noise_map}
                path_to_data: {self.path_to_data}
                mean_amp_snr: {self.mean_amp_snr}
                std_amp_snr: {self.std_amp_snr}
                mean_linewidth: {self.mean_linewidth}
                std_linewidth: {self.std_linewidth}
                mean_ncomponent: {self.mean_ncomponent}
                std_ncomponent: {self.std_ncomponent}
                fix_velocities: {self.fix_velocities}
                fix_velocities_sigma: {self.fix_velocities_sigma}
                smooth_testdata: {self.smooth_testdata}
                lam1: {self.lam1}
                p1: {self.p1}
                lam2: {self.lam2}
                p2: {self.p2}
                niters: {self.niters}
                iterations_for_convergence: {self.iterations_for_convergence}
                noise: {self.noise}
                add_residual: {self.add_residual}
                sig: {self.sig}
                velo_range: {self.velo_range}
                check_signal_sigma: {self.check_signal_sigma}
                p_limit: {self.p_limit}
                ncpus: {self.ncpus}
                suffix: {self.suffix}
                filename_out: {self.filename_out}
                seed: {self.seed}
                )'''
    
    def getting_ready(self):
        """
        Prints a message when preparation starts.

        """
        string = 'preparation'
        banner = len(string) * '='
        heading = '\n' + banner + '\n' + string + '\n' + banner
        say(heading)

    def prepare_data(self):
        """
        Prepares the training data by reading in data and
        setting up lists for Gaussian parameter distributions.

        """
        self.getting_ready()
        self.image = fits.getdata(self.fitsfile) #load data
        self.image[np.where(np.isnan(self.image))] = 0.0

        self.header = fits.getheader(self.fitsfile)
        #self.header_2d = md_header_2d(self.fitsfile)
        self.v = self.header['NAXIS3']
        self.velocity = velocity_axes(self.fitsfile)
        self.mock_data = {'training_data' : None, 'test_data' : None, 'hisa_spectra' : None, 'location' : None, 'amplitudes' : None, 'fwhms' : None, 'means' : None, 'hisa_mask' : None, 'signal_ranges' : None, 'rms_noise' : None, 'velocity' : None, 'header' : None}
        self.hisa_spectra = []
        self.training_data = []
        self.test_data = []
        self.location = []
        self.amplitudes = []
        self.fwhms = []
        self.means = []
        self.hisa_mask = []
        self.signal_ranges = []
        string = 'Done!'
        say(string)

    def prepare_training(self):
        """
        Creates the test and training data and saves them into a pickled file.
        
        """
        self.rng = np.random.default_rng(self.seed)
        self.prepare_data()

        if self.training_set_size <= 0:
            raise ValueError("'training_set_size' has to be >0")
        if self.lam1 is None:
            self.lam1 = 2.00
        if self.p1 is None:
            self.p1 = 0.90
        if not 0<= self.p1 <=1:
            raise ValueError("'p1' has to be in the range [0,1]")
        if self.lam2 is None:
            self.lam2 = 2.00
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
            self.p_limit = 0.01
            
        string = 'creating training data'
        banner = len(string) * '='
        heading = '\n' + banner + '\n' + string + '\n' + banner
        say(heading)

        self.max_consec_ch = get_max_consecutive_channels(self.v, self.p_limit)
        channel_width = self.header['CDELT3'] / 1000.
        spectral_resolution = 1 / np.sqrt(8*np.log(2)) # unit channel
        edges = int(0.2 * min(self.header['NAXIS1'],self.header['NAXIS2']))
        indices = np.column_stack((self.rng.integers(edges,self.header['NAXIS2']-edges+1,self.training_set_size), self.rng.integers(edges,self.header['NAXIS1']-edges+1,self.training_set_size)))

        if self.mean_linewidth is None or self.std_linewidth is None:
            raise ValueError('No linewidth parameters are given to create test data.')
        mu_lws_HISA, sigma_lws_HISA = (self.mean_linewidth / channel_width) / np.sqrt(8*np.log(2)), self.std_linewidth / channel_width # mean and standard deviation
        # TODO
        if self.fix_velocities is None:
            mu_ncomps_HISA, sigma_ncomps_HISA = self.mean_ncomponent, self.std_ncomponent
            ncomps_HISA = np.around(self.rng.normal(mu_ncomps_HISA, sigma_ncomps_HISA, self.training_set_size).reshape(self.training_set_size)).astype(int)
            ncomps_HISA[ncomps_HISA<=0] = int(1.)
        else:
            fix_velocities_indices = np.array([find_nearest(self.velocity,e) for e in self.fix_velocities]) 
            ncomps_HISA = np.full(self.training_set_size, len(fix_velocities_indices)).reshape(self.training_set_size).astype(int)

        xvals = np.arange(0,self.v,1)
        
        self.spectrum_list = []
        self.location = []
        self.noise_list = []
        self.thresh_list = []
        for idx, (y, x) in enumerate(zip(indices[:,0], indices[:,1])):
            self.spectrum_list.append(self.image[:,y,x])
            self.location.append((y,x))
            self.noise_list.append(noise_map[y,x])
            self.thresh_list.append(thresh[y,x])
        import astrosaber.parallel_processing
        astrosaber.parallel_processing.init([self.spectrum_list, [self]])
        #ilist = np.arange(len(self.spectrum_list))
        results_list = astrosaber.parallel_processing.func(use_ncpus=self.ncpus, function='hisa') # initiate parallel process
        # sort results from parallel process by original indices to keep the same order (not needed for single core use)
        results_list.sort(key=lambda x: x[5])
        for i in trange(len(results_list)):
            amp_list = []
            fwhm_list = []
            mean_list = []
            #Check for NaNs in the test spectra
            if np.any(np.isnan(results_list[i][0])):
                print('Mock spectrum contains NaN! Will remove it!')
                continue
            samplesize_rng = 50 * ncomps_HISA[i]
            amps_HISA = self.rng.normal(results_list[i][3], results_list[i][4], samplesize_rng).reshape(samplesize_rng,) # self.training_set_size
            amps_HISA[amps_HISA<3.*self.noise_list[i]] = 3*self.noise_list[i]
            velos_of_comps_HISA = []
            for j in range(ncomps_HISA[i]):
                if self.fix_velocities is None:
                    k = 0
                    mu_velos_HISA_k, sigma_velos_HISA_k = (results_list[i][1][k,0] + results_list[i][1][k,1]) / 2., (results_list[i][1][k,1] - results_list[i][1][k,0]) / (np.sqrt(8*np.log(2))) / 3. # mean and standard deviation
                    velos_HISA_k = self.rng.normal(mu_velos_HISA_k, sigma_velos_HISA_k, samplesize_rng).reshape(samplesize_rng,)
                    # limit hisa signal ranges
                    signal_mask = np.logical_and(velos_HISA_k>results_list[i][1][k,0], velos_HISA_k<results_list[i][1][k,1])
                    velos_HISA_k = velos_HISA_k[signal_mask]
                    #velos_HISA_k = self.rng.integers(low=results_list[i][1][k,0], high=results_list[i][1][k,1], endpoint=True, size=samplesize_rng)
                    if k < len(results_list[i][1][:,0])-1:
                        k += 1
                else:
                    if self.fix_velocities_sigma is not None:
                        fix_velocities_sigma_k = self.fix_velocities_sigma / channel_width
                    else:
                        fix_velocities_sigma_k = 1. # one spectral channel as std
                    velos_HISA_k = self.rng.normal(fix_velocities_indices[j], fix_velocities_sigma_k, samplesize_rng).reshape(samplesize_rng,)
                velos_of_comps_HISA_k = self.rng.choice(velos_HISA_k, 1)
                if not (velos_of_comps_HISA_k < 0 or velos_of_comps_HISA_k > self.v):
                    velos_of_comps_HISA.append(velos_of_comps_HISA_k)
            velos_of_comps_HISA = np.array(velos_of_comps_HISA) 
            lws_HISA = self.rng.normal(mu_lws_HISA, sigma_lws_HISA, samplesize_rng).reshape(samplesize_rng,) # 
            amps_of_comps_HISA = self.rng.choice(amps_HISA, ncomps_HISA[i])
            lws_of_comps_HISA = self.rng.choice(lws_HISA, ncomps_HISA[i])  
            ncomp_HISA = np.arange(0,ncomps_HISA[i]+1,1)
            lws_of_comps_HISA[np.where(lws_of_comps_HISA<spectral_resolution)] = spectral_resolution

            gauss_HISA = np.zeros(shape=(self.v,))
            ranges_hisa_list = []
            for idx, (v, lw, amp) in enumerate(zip(velos_of_comps_HISA,lws_of_comps_HISA,amps_of_comps_HISA)):
                exp_arg = 0.5 * ((xvals - v) / lw)**2
                exp_arg[np.where(exp_arg>100.)] = 100.
                #limit HISA to 3sigma of HI emission
                if results_list[i][0][int(np.around(v))] - amp < 3*self.noise_list[i]:
                    amp = results_list[i][0][int(np.around(v))] - 3*self.noise_list[i]
                #if amp>results_list[i][0][int(np.around(v))]:
                #    amp = results_list[i][0][int(np.around(v))]
                gauss_HISA = gauss_HISA + amp * np.exp(-exp_arg)
                ranges_hisa_i = [np.around(v - 3*lw), np.around(v + 3*lw)]
                ranges_hisa_list.append(ranges_hisa_i)
                amp_list.append(amp)
                fwhm_list.append(lw * channel_width * np.sqrt(8*np.log(2)))
                mean_list.append(((self.header['CRVAL3'] - self.header['CRPIX3'] * self.header['CDELT3']) + (v+1) * self.header['CDELT3']) / 1000.)
            gauss_HISA[np.where(gauss_HISA<1e-5)] = 0.
                
            ranges_hisa = np.array(ranges_hisa_list).astype(int).reshape(-1,2)
            sort_indices = np.argsort(ranges_hisa[:, 0])
            ranges_hisa = ranges_hisa[sort_indices]
            consecutive_channels_hisa = ranges_hisa[:, 1] - ranges_hisa[:, 0]
            mask_ranges_hisa = ranges_hisa[np.where(consecutive_channels_hisa>=0)]
            pad = 2
            for j in range(mask_ranges_hisa.shape[0]):
                lower = max(0, mask_ranges_hisa[j,0] - pad)
                upper = min(self.v, mask_ranges_hisa[j,1] + pad)
                mask_ranges_hisa[j,0], mask_ranges_hisa[j,1] = lower, upper
            mask_hisa = mask_channels(self.v, mask_ranges_hisa, pad_channels=pad, remove_intervals=None)
            #mask HISA where no HI emission
            gauss_HISA[np.invert(results_list[i][2])] = 0.   
            
            self.training_data.append(results_list[i][0] - gauss_HISA)
            self.test_data.append(results_list[i][0])
            self.hisa_spectra.append(gauss_HISA)
            self.amplitudes.append(amp_list)
            self.fwhms.append(fwhm_list)
            self.means.append(mean_list)
            self.hisa_mask.append(mask_hisa)
            self.signal_ranges.append(mask_ranges_hisa)

        self.mock_data['training_data'] = self.training_data
        self.mock_data['test_data'] = self.test_data
        self.mock_data['hisa_spectra'] = self.hisa_spectra
        self.mock_data['location'] = self.location
        self.mock_data['amplitudes'] = self.amplitudes
        self.mock_data['fwhms'] = self.fwhms
        self.mock_data['means'] = self.means
        self.mock_data['hisa_mask'] = self.hisa_mask
        self.mock_data['signal_ranges'] = self.signal_ranges
        self.mock_data['rms_noise'] = self.noise_list
        self.mock_data['velocity'] = self.velocity
        self.mock_data['header'] = self.header

        self.save_data()
        plot_pickle_spectra(self.path_to_file, outfile=None, ranges=None, path_to_plots='astrosaber_training/plots', n_spectra=20, rowsize=4., rowbreak=10, dpi=72, velocity_range=[self.velocity[0],self.velocity[-1]], vel_unit=u.km/u.s, seed=self.seed)

    def two_step_extraction_prepare(self, i : int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Runs the two-phase smoothing with default parameters to generate test data and self-absorption parameters.
        
        Returns
        -------
        mock_emission : numpy.ndarray
            'Pure' emission spectrum w/o self-absorption.
        mask_ranges : numpy.ndarray
            Array of range indices that contain signal.
        mask : numpy.ndarray
            Array of signal mask.
        mu_amps_HISA : float
            Mean amplitude value of self-absorption features.
        sigma_amps_HISA : float
            Standard deviation of self-absorption features.
        """
        bg, _, _, _  = two_step_extraction(self.lam1, self.p1, self.lam2, self.p2, spectrum=self.spectrum_list[i], header=self.header, check_signal_sigma=self.check_signal_sigma, noise=self.noise_list[i], velo_range=self.velo_range, niters=self.niters, iterations_for_convergence=self.iterations_for_convergence, add_residual=self.add_residual, thresh=self.thresh_list[i])
        
        pad_ch = 5
        consecutive_channels, ranges = determine_peaks(spectrum=self.spectrum_list[i], peak='positive', amp_threshold=None)
        mask_ranges = ranges[np.where(consecutive_channels>=self.max_consec_ch+2*pad_ch)]
        mask = mask_channels(self.v, mask_ranges, pad_channels=-1*pad_ch, remove_intervals=None)
        
        obs_noise = self.rng.normal(0,self.noise_list[i],size=(self.v,))
        if self.smooth_testdata is True:
            mock_emission = bg + obs_noise
        elif self.smooth_testdata is False:
            mock_emission = self.spectrum_list[i]

        mu_amps_HISA, sigma_amps_HISA = self.mean_amp_snr*self.noise_list[i], self.std_amp_snr*self.noise_list[i]

        return mock_emission, mask_ranges, mask, mu_amps_HISA, sigma_amps_HISA, i

    def save_data(self):
        """
        Saves all the test and training data into a pickled file.
        
        """
        if self.filename_out is None:
            filename_wext = os.path.basename(self.fitsfile)
            filename_base, file_extension = os.path.splitext(filename_wext)
            filename_out = '{}-training_set-{}_spectra{}.pickle'.format(filename_base, self.training_set_size, self.suffix)
        elif not self.filename_out.endswith('.pickle'):
            filename_out = self.filename_out + '.pickle'
        else:
            filename_out = self.filename_out
        dirname = os.path.join(self.path_to_data, 'astrosaber_training')
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.path_to_file = os.path.join(dirname, filename_out)
        pickle.dump(self.mock_data, open(self.path_to_file, 'wb'), protocol=2)
        say("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(filename_out, dirname))
