'''asymmetric least squares fit'''

import os
import numpy as np

from astropy.io import fits
from scipy import sparse
from scipy.sparse.linalg import spsolve

from tqdm import trange
import warnings

from .quality_checks import goodness_of_fit, get_max_consecutive_channels, determine_peaks, mask_channels
from .aslsq_helper import check_signal_ranges, count_ones_in_row, IterationWarning, say, format_warning

warnings.showwarning = format_warning


#Asymmetric least squares baseline fit from Eilers et al. 2005
def baseline_als(y, lam, p, niter):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


#optimized version; this should be faster by a factor ~1.5
def baseline_als_optimized(y, lam, p, niter):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) 
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


def one_step_extraction(lam1, p1, spectrum=None, header=None, check_signal_sigma=6., noise=None, velo_range=15.0, niters=50, iterations_for_convergence=3, add_residual=False, thresh=None):
    flag_map = 1.
    if check_signal_ranges(spectrum, header, sigma=check_signal_sigma, noise=noise, velo_range=velo_range):
        spectrum_prior = baseline_als_optimized(spectrum, lam1, p1, niter=3)
        spectrum_firstfit = spectrum_prior
        converge_logic = np.array([])
        for n in range(niters+1):
            spectrum_prior = baseline_als_optimized(spectrum_prior, lam1, p1, niter=3)
            spectrum_next = baseline_als_optimized(spectrum_prior, lam1, p1, niter=3)
            residual = abs(spectrum_next - spectrum_prior)
            if np.any(np.isnan(residual)):
                print('Residual contains NaNs') 
                residual[np.isnan(residual)] = 0.0
            if thresh==0.:
                converge_test = (True)
            else:
                converge_test = (np.all(residual < thresh))
            converge_logic = np.append(converge_logic,converge_test)
            c = count_ones_in_row(converge_logic)
            if np.any(c > iterations_for_convergence):
                i_converge = np.min(np.argwhere(c > iterations_for_convergence))
                res = abs(spectrum_next - spectrum_firstfit)
                if add_residual:
                    final_spec = spectrum_next + res
                else:
                    final_spec = spectrum_next
                break
            elif n==niters:
                warnings.warn('Maximum number of iterations reached. Fit did not converge.', IterationWarning)
                #flags
                flag_map = 0.
                res = abs(spectrum_next - spectrum_firstfit)
                if add_residual:
                    final_spec = spectrum_next + res
                else:
                    final_spec = spectrum_next
                i_converge = niters
                break
        bg = final_spec - thresh
        hisa = final_spec - spectrum - thresh
        iterations = i_converge
    else:
        bg = np.full_like(spectrum, np.nan)
        hisa = np.full_like(spectrum, np.nan)
        iterations = np.nan
        #flags
        flag_map = 0.
    return bg, hisa, iterations, flag_map


def two_step_extraction(lam1, p1, lam2, p2, spectrum=None, header=None, check_signal_sigma=6., noise=None, velo_range=15.0, niters=50, iterations_for_convergence=3, add_residual=True, thresh=None):
    flag_map = 1.
    if check_signal_ranges(spectrum, header, sigma=check_signal_sigma, noise=noise, velo_range=velo_range):
        spectrum_prior = baseline_als_optimized(spectrum, lam1, p1, niter=3)
        spectrum_firstfit = spectrum_prior
        converge_logic = np.array([])
        for n in range(niters+1):
            spectrum_prior = baseline_als_optimized(spectrum_prior, lam2, p2, niter=3)
            spectrum_next = baseline_als_optimized(spectrum_prior, lam2, p2, niter=3)
            residual = abs(spectrum_next - spectrum_prior)
            if np.any(np.isnan(residual)):
                print('Residual contains NaNs') 
                residual[np.isnan(residual)] = 0.0
            if thresh==0.:
                converge_test = (True)
            else:
                converge_test = (np.all(residual < thresh))
            converge_logic = np.append(converge_logic,converge_test)
            c = count_ones_in_row(converge_logic)
            if np.any(c > iterations_for_convergence):
                i_converge = np.min(np.argwhere(c > iterations_for_convergence))
                res = abs(spectrum_next - spectrum_firstfit)
                if add_residual:
                    final_spec = spectrum_next + res
                else:
                    final_spec = spectrum_next
                break
            elif n==niters:
                warnings.warn('Maximum number of iterations reached. Fit did not converge.', IterationWarning)
                #flags
                flag_map = 0.
                res = abs(spectrum_next - spectrum_firstfit)
                if add_residual:
                    final_spec = spectrum_next + res
                else:
                    final_spec = spectrum_next
                i_converge = niters
                break
        # TODO
        consecutive_channels, ranges = determine_peaks(spectrum, peak='both', amp_threshold=None)
        mask_ranges = ranges[np.where(consecutive_channels>=max_consec_ch)]
        mask = mask_channels(len(spectrum), mask_ranges, pad_channels=2, remove_intervals=None)
        if np.all(mask):
            noise_fit_offset = thresh
        else:
            noise_range = np.invert(mask)
            noise_fit_offset = np.nanmean(final_spec[noise_range])
        #
        bg = final_spec - noise_fit_offset
        hisa = final_spec - spectrum - noise_fit_offset
        iterations = i_converge
    else:
        bg = np.full_like(spectrum, np.nan)
        hisa = np.full_like(spectrum, np.nan)
        iterations = np.nan
        #flags
        flag_map = 0.
    return bg, hisa, iterations, flag_map
