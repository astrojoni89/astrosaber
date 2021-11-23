import sys
import numpy as np
import pickle
from astropy.io import fits
from scipy import sparse
from scipy.sparse.linalg import spsolve

from tqdm import tqdm, trange
from tqdm.utils import _is_utf, _supports_unicode
import warnings
import os

from .utils.quality_checks import goodness_of_fit, get_max_consecutive_channels, determine_peaks, mask_channels
from .utils.aslsq_helper import velocity_axes, count_ones_in_row, check_signal_ranges, IterationWarning, say, format_warning
from .utils.aslsq_fit import baseline_als_optimized, two_step_extraction

warnings.showwarning = format_warning



class saberTraining(object):
    def __init__(self, pickle_file, path_to_data='.', iterations=100, lam1_initial=None, p1=None, lam2_initial=None, p2=None, weight_1=None, weight_2=None, lam1_bounds=None, lam2_bounds=None, MAD=None, eps=None, learning_rate=None, mom=None, get_trace=False, niters=20, iterations_for_convergence=3, add_residual = True, sig = 1.0, velo_range = 15.0, check_signal_sigma = 6., p_limit=None, ncpus=None, suffix='', filename_out=None, seed=111):
        self.pickle_file = pickle_file
        self.path_to_data = path_to_data

        self.iterations = iterations
        
        self.lam1_initial = lam1_initial
        self.p1 = p1
        self.lam2_initial = lam2_initial
        self.p2 = p2

        self.weight_1 = weight_1
        self.weight_2 = weight_2
        self.lam1_bounds = lam1_bounds
        self.lam2_bounds = lam2_bounds
        self.MAD = MAD
        self.eps = eps
        self.learning_rate = learning_rate
        self.mom = mom
        self.get_trace = get_trace
        
        self.niters = int(niters)
        self.iterations_for_convergence = int(iterations_for_convergence)
        
        self.add_residual = add_residual
        self.sig = sig
        
        self.velo_range = velo_range
        self.check_signal_sigma = check_signal_sigma

        self.p_limit = p_limit

        self.ncpus = ncpus

        self.suffix = suffix
        self.filename_out = filename_out
        
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
      
    def __str__(self):
        return f'saberTraining:\npickle_file: {self.pickle_file}\npath_to_data: {self.path_to_data}\niterations: {self.iterations}\nlam1_initial: {self.lam1_initial}\np1: {self.p1}\nlam2_initial: {self.lam2_initial}\np2: {self.p2}\nweight_1: {self.weight_1}\nweight_2: {self.weight_2}\nlam1_bounds: {self.lam1_bounds}\nlam2_bounds: {self.lam2_bounds}\nMAD: {self.MAD}\neps: {self.eps}\nlearning_rate: {self.learning_rate}\nmom: {self.mom}\nget_trace: {self.get_trace}\nniters: {self.niters}\niterations_for_convergence: {self.iterations_for_convergence}\nadd_residual: {self.add_residual}\nsig: {self.sig}\nvelo_range: {self.velo_range}\ncheck_signal_sigma: {self.check_signal_sigma}\np_limit: {self.p_limit}\nncpus: {self.ncpus}\nsuffix: {self.suffix}\nfilename_out: {self.filename_out}\nseed: {self.seed}'

    def getting_ready(self):
        string = 'preparation'
        banner = len(string) * '='
        heading = '\n' + banner + '\n' + string + '\n' + banner
        say(heading)

    def prepare_data(self):
        self.getting_ready()
        self.p = pickle.load(open(self.pickle_file, 'rb'), encoding='latin1')
        self.training_data = self.p['training_data']
        self.test_data = self.p['test_data']
        self.noise = np.array(self.p['rms_noise'])
        self.thresh = self.sig * self.noise
        self.velocity = self.p['velocity']
        self.header = self.p['header']
        self.v = len(self.p['velocity'])
        if self.p_limit is None:
            self.p_limit = 0.02
        if self.p1 is None:
            self.p1 = 0.90
        if self.p2 is None:
            self.p2 = 0.90
        if self.weight_1 is None:
            self.weight_1 = 0.0
        if self.weight_2 is None:
            self.weight_2 = 0.0
        self.max_consec_ch = get_max_consecutive_channels(self.v, self.p_limit)
        #init(self.training_data)
        #self.ilist = np.arange(len(self.training_data))
        string = 'Done!'
        say(string)

    def training(self):
        self.prepare_data()
        string = 'Optimizing smoothing parameters'
        banner = len(string) * '='
        heading = '\n' + banner + '\n' + string + '\n' + banner
        say(heading)
        self.popt_lam = self.train()
        self.save_data()

    def train(self):
        popt_lam = self.train_lambda_set(self.objective_function_lambda_set, training_data=self.training_data, test_data=self.test_data, noise=self.noise, lam1_initial=self.lam1_initial, p1=self.p1, lam2_initial=self.lam2_initial, p2=self.p2, lam1_bounds=self.lam1_bounds, lam2_bounds=self.lam2_bounds, iterations=self.iterations, MAD=self.MAD, eps=self.eps, learning_rate=self.learning_rate, mom=self.mom, window_size=5, iterations_for_convergence_training=10, ncpus=self.ncpus)
        return popt_lam

    def objective_function_lambda_set(self, lam1, p1, lam2, p2, get_all=True, dof=4, ncpus=None): #, training_data=None, test_data=None, header=None, check_signal_sigma=None, noise=None, velo_range=None, niters=None, iterations_for_convergence=None, add_residual=None, thresh=None, mask=None, get_all=True, dof=4, 
   
        #global lam1_updt, lam2_updt
        self.lam1_updt, self.lam2_updt = lam1, lam2
        import astroSABER.parallel_processing
        astroSABER.parallel_processing.init([self.training_data, [self]])
        results_list = np.array(astroSABER.parallel_processing.func_wo_bar(use_ncpus=ncpus, function='cost'))
   
        if get_all:
            return np.nanmedian(results_list[:,0]), np.nanmedian(results_list[:,1]), np.nanmedian(results_list[:,2]) #gmean(cost_function_list),  gmean(rchi2_list)
        else:
            return np.nanmedian(results_list[:,0]) #gmean(cost_function_list)

    def single_cost(self, i, get_all=True, dof=4): # , lam1_updt=None, p1_updt=None, lam2_updt=None, p2_updt=None
        consecutive_channels, ranges = determine_peaks(self.training_data[i], peak='both', amp_threshold=None)
        mask_ranges = ranges[np.where(consecutive_channels>=self.max_consec_ch)]
        mask = mask_channels(self.v, mask_ranges, pad_channels=2, remove_intervals=None)

        bg_fit, _, _, _ = two_step_extraction(self.lam1_updt, self.p1, self.lam2_updt, self.p2, spectrum=self.training_data[i], header=self.header, check_signal_sigma=self.check_signal_sigma, noise=self.noise[i], velo_range=self.velo_range, niters=self.niters, iterations_for_convergence=self.iterations_for_convergence, add_residual=self.add_residual, thresh=self.thresh[i])
    
        if type(self.noise[i]) is not np.ndarray:
            noise_array = np.ones(len(self.training_data[i])) * self.noise[i]
        else:
            noise_array = self.noise[i]
        if mask is None:
            mask = np.ones(len(self.training_data[i]))
            mask = mask.astype('bool')
        elif len(mask) == 0:
            mask = np.ones(len(self.training_data[i]))
            mask = mask.astype('bool')
        elif np.count_nonzero(mask) == 0:
            mask = np.ones(len(self.training_data[i]))
            mask = mask.astype('bool')
        if any(np.isnan(bg_fit)):
            bg_fit = np.full((len(self.training_data[i])), np.nan)
            warnings.warn('Asymmetric least squares fit contains NaNs.', IterationWarning)
    
        squared_residuals = (self.test_data[i][mask] - bg_fit[mask])**2
        residuals = (self.test_data[i][mask] - bg_fit[mask])
        ssr = np.nansum(squared_residuals)
        cost_function = ssr / (2 * len(self.test_data[i][mask])) + self.weight_1 * self.lam1_updt + self.weight_2 * self.lam2_updt #penalize large smoothing
        if get_all:    
            chi2 = np.nansum(squared_residuals / noise_array[mask]**2)
            n_samples = len(self.test_data[i][mask])
            rchi2 = chi2 / (n_samples - dof)
            MAD = np.nansum(abs(residuals)) / n_samples
            return cost_function, rchi2, MAD
        else:
            return cost_function

    class gradient_descent_lambda_set(object):
        """Bookkeeping object."""
        def __init__(self, iterations):
            self.lam1_trace = np.zeros(iterations+1) * np.nan
            self.lam2_trace = np.zeros(iterations+1) * np.nan
            self.accuracy_trace = np.zeros(iterations) * np.nan
            self.D_lam1_trace = np.zeros(iterations) * np.nan
            self.D_lam2_trace = np.zeros(iterations) * np.nan
            self.lam1means1 = np.zeros(iterations) * np.nan
            self.lam1means2 = np.zeros(iterations) * np.nan
            self.lam2means1 = np.zeros(iterations) * np.nan
            self.lam2means2 = np.zeros(iterations) * np.nan
            self.fracdiff_lam1 = np.zeros(iterations) * np.nan
            self.fracdiff_lam2 = np.zeros(iterations) * np.nan
            self.iter_of_convergence = np.nan

    def train_lambda_set(self, objective_function, training_data=None, test_data=None, noise=None, lam1_initial=None, p1=None, lam2_initial=None, p2=None, lam1_bounds=None, lam2_bounds=None, iterations=100, MAD=None, eps=None, learning_rate=None, mom=None, window_size=5, iterations_for_convergence_training=10, get_trace=False, mask=None, ncpus=None):
        """
        lam1_initial =
        lam2_initial =
        iterations =
        MAD = mean absolute difference
        eps = 'epsilson; finite offset for computing derivatives in gradient'
        learning_rate
        mom = 'Momentum value'
        window_size = trailing window size to determine convergence,
        iterations_for_convergence_training = number of continuous iterations within threshold tolerence required to achieve convergence
        """

        # Default settings for hyper parameters
        if self.learning_rate is None:
            self.learning_rate = 0.01
        if self.eps is None:
            self.eps = 0.01
        if self.MAD is None:
            self.MAD = 0.03
        if self.mom is None:
            self.mom = .1

        tolerance = self.MAD / np.sqrt(window_size)

        if self.lam1_initial is None:
            raise ValueError("'lam1_initial' parameter is required for training.")
        if self.lam2_initial is None:
            raise ValueError("'lam2_initial' parameter is required for training.")

        if self.lam1_initial <= self.lam2_initial:
            raise ValueError("'lam1_initial' has to be greater than 'lam2_initial'")

        # Initialize book-keeping object
        gd = self.gradient_descent_lambda_set(self.iterations)
        gd.lam1_trace[0] = self.lam1_initial
        gd.lam2_trace[0] = self.lam2_initial

        for i in range(self.iterations):
            self.lam1_r, self.lam1_c, self.lam1_l = gd.lam1_trace[i] + self.eps, gd.lam1_trace[i], gd.lam1_trace[i] - self.eps
            self.lam2_r, self.lam2_c, self.lam2_l = gd.lam2_trace[i] + self.eps, gd.lam2_trace[i], gd.lam2_trace[i] - self.eps
        

            # Calls to objective function
            #lam1
            obj_lam1r, rchi2_lam1r, _ = objective_function(self.lam1_r, self.p1, self.lam2_c, self.p2, ncpus=self.ncpus) # self.training_data, self.test_data, self.header, self.check_signal_sigma, self.noise, self.velo_range, self.niters, self.iterations_for_convergence, self.add_residual, self.thresh, self.mask, get_all=True, dof=4,
            obj_lam1l, rchi2_lam1l, _ = objective_function(self.lam1_l, self.p1, self.lam2_c, self.p2, ncpus=self.ncpus)
            gd.D_lam1_trace[i] = (obj_lam1r - obj_lam1l) / 2. / self.eps

            #lam2
            obj_lam2r, rchi2_lam2r, _ = objective_function(self.lam1_c, self.p1, self.lam2_r, self.p2, ncpus=self.ncpus)
            obj_lam2l, rchi2_lam2l, _ = objective_function(self.lam1_c, self.p1, self.lam2_l, self.p2, ncpus=self.ncpus)
            gd.D_lam2_trace[i] = (obj_lam2r - obj_lam2l) / 2. / self.eps

            gd.accuracy_trace[i] =  (rchi2_lam1r + rchi2_lam1l + rchi2_lam2r + rchi2_lam2l) / 4.

            if i == 0:
                momentum_lam1, momentum_lam2 = 0., 0. #, momentum_lam2, momentum_p2 
            else:
                if self.mom < 0 or self.mom > 1:
                    raise ValueError("'mom' must be between zero and one")

                momentum_lam1 = self.mom * (gd.lam1_trace[i] - gd.lam1_trace[i-1])
                momentum_lam2 = self.mom * (gd.lam2_trace[i] - gd.lam2_trace[i-1])

            gd.lam1_trace[i+1] = gd.lam1_trace[i] - self.learning_rate * gd.D_lam1_trace[i] + momentum_lam1
            gd.lam2_trace[i+1] = gd.lam2_trace[i] - self.learning_rate * gd.D_lam2_trace[i] + momentum_lam2

        # lam and p cannot be negative; p needs to be 0<p<1
            if self.lam1_bounds is None:
                self.lam1_bounds = [0.1,5.0]
            if gd.lam1_trace[i+1] < min(self.lam1_bounds):
                gd.lam1_trace[i+1] = min(self.lam1_bounds) + 0.05
            if gd.lam1_trace[i+1] > max(self.lam1_bounds):
                gd.lam1_trace[i+1] = max(self.lam1_bounds) - 0.05
            if self.lam2_bounds is None:
                self.lam2_bounds = [0.1,5.0]
            if gd.lam2_trace[i+1] < min(self.lam2_bounds):
                gd.lam2_trace[i+1] = min(self.lam2_bounds) + 0.05
            if gd.lam2_trace[i+1] > max(self.lam2_bounds):
                gd.lam2_trace[i+1] = max(self.lam2_bounds) - 0.05
        
            if gd.lam1_trace[i+1] < 0.:
                gd.lam1_trace[i+1] = 0.
            if gd.lam2_trace[i+1] < 0.:
                gd.lam2_trace[i+1] = 0.
            lam1_bound_i = gd.lam2_trace[i+1]
            if gd.lam1_trace[i+1] <= lam1_bound_i:
                gd.lam1_trace[i+1] = gd.lam2_trace[i+1] + 0.05

            say('\niter {0}: red.chi square={1:4.2f}, [lam1, lam2]=[{2:.3f}, {3:.3f}], [p1, p2]=[{4:.3f}, {5:.3f}], mom=[{6:.1f}, {7:4.1f}]'.format(i, gd.accuracy_trace[i], np.round(gd.lam1_trace[i], 3), np.round(gd.lam2_trace[i], 3), np.round(p1, 3), np.round(p2, 3), np.round(momentum_lam1, 2), np.round(momentum_lam2, 2)), end=' ')


    #    if False: (use this to avoid convergence testing)
            if i <= 2 * window_size:
                say(' (Convergence testing begins in {} iterations)'.format(int(2 * window_size - i)))
            else:
                gd.lam1means1[i] = np.mean(gd.lam1_trace[i - window_size:i])
                gd.lam1means2[i] = np.mean(gd.lam1_trace[i - 2 * window_size:i - window_size])
          
                gd.lam2means1[i] = np.mean(gd.lam2_trace[i - window_size:i])
                gd.lam2means2[i] = np.mean(gd.lam2_trace[i - 2 * window_size:i - window_size])

                gd.fracdiff_lam1[i] = np.abs(gd.lam1means1[i] - gd.lam1means2[i])
                gd.fracdiff_lam2[i] = np.abs(gd.lam2means1[i] - gd.lam2means2[i])

                converge_logic = (gd.fracdiff_lam1 < tolerance) & (gd.fracdiff_lam2 < tolerance)

                c = count_ones_in_row(converge_logic)
                say('  ({0:4.3F},{1:4.3F} < {2:4.3F} for {3} iters [{4} required])'.format(gd.fracdiff_lam1[i], gd.fracdiff_lam2[i], tolerance, int(c[i]), iterations_for_convergence_training))

                if i in range(2 * window_size, self.iterations, 10):
                    if _supports_unicode(sys.stderr):
                        quote = '    "Much to learn, you still have!"      '
                        offset = ' ' * int(len(quote))
                        print('\n' + quote + '    ﹏    ') 
                        print(offset + '<´(⬬ ⬬)`> ')
                        print(offset + ' ʿ/   \ʾ  ')

                if np.any(c > iterations_for_convergence_training):
                    i_converge_training = np.min(np.argwhere(c > iterations_for_convergence_training))
                    gd.iter_of_convergence = i_converge_training
                    say('\nStable convergence achieved at iteration: {}'.format(i_converge_training))
                    break

        # Return best-fit lambdas, and bookkeeping object
        if self.get_trace:
            return gd.lam1_trace, gd.lam2_trace
        return gd.lam1means1[i], gd.lam2means1[i]
    
    def save_data(self):
        if self.filename_out is None:
            if not self.get_trace:
                filename_lam = self.pickle_file.split('/')[-1].split('.pickle')[0]+'_lam_opt.txt'
            else:
                filename_lam = self.pickle_file.split('/')[-1].split('.pickle')[0]+'_lam_traces.txt'
        else:
            filename_lam = str(self.filename_out) + '.txt'
        pathname_lam = os.path.join(self.path_to_data, filename_lam)
        np.savetxt(pathname_lam, self.popt_lam)
        print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(filename_lam, self.path_to_data))
