'''asymmetric least squares fit'''

import os
import numpy as np
from typing import Tuple

from astropy.io import fits
from scipy import sparse
from scipy.sparse.linalg import spsolve

from tqdm import trange
import warnings

from .quality_checks import get_max_consecutive_channels, determine_peaks, mask_channels
from .aslsq_helper import check_signal_ranges, count_ones_in_row, IterationWarning, say, format_warning

warnings.showwarning = format_warning


#Asymmetric least squares baseline fit from Eilers et al. 2005
def baseline_als(y : np.ndarray, lam : float, p : float, niter : int) -> np.ndarray:
    """
    Baseline smoothing using asymmetric least squares.

    Parameters
    ----------
    y : numpy.ndarray
        Spectrum to be smoothed.
    lam : float
        Smoothing weight. Adjusts the amount of smoothing.
    p : float
        Asymmetry weight. Adjusts how much weight positive or negative signals (wrt the smoothed baseline) will be given.
    niter : int
        Number of iterations.

    Returns
    -------
    z : numpy.ndarray
        Smoothed baseline.
    """
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
def baseline_als_optimized(y : np.ndarray, lam : float, p : float, niter : int, mask : np.ndarray = None) -> np.ndarray:
    """
    Baseline smoothing using asymmetric least squares.

    Parameters
    ----------
    y : numpy.ndarray
        Spectrum to be smoothed.
    lam : float
        Smoothing weight. Adjusts the amount of smoothing.
    p : float
        Asymmetry weight. Adjusts how much weight positive or negative signals (wrt the smoothed baseline) will be given.
    niter : int
        Number of iterations.
    mask : numpy.ndarray
        Boolean mask indicating signal ranges, with `True` indicating signal, `False` indicating noise.
        Will be used to set asymmetry weights to 0.5 where there is noise.

    Returns
    -------
    z : numpy.ndarray
        Smoothed baseline.
    """
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
        if mask is not None:
            w[np.invert(mask)] = 0.5
    return z


def one_step_extraction(lam1 : float, p1 : float, spectrum : np.ndarray = None, header : fits.Header = None,
                        check_signal_sigma : float = 6., noise : float = None, velo_range : float = 15.0,
                        niters : int = 20, iterations_for_convergence : int = 3, add_residual : bool = False,
                        thresh : float = None, p_limit : float = 0.02) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Baseline smoothing routine using one lambda smoothing value for all major iterations.

    Parameters
    ----------
    lam1 : float
        Smoothing weight. Adjusts the amount of smoothing.
    p1 : float
        Asymmetry weight. Adjusts how much weight positive or negative signals (wrt the smoothed baseline) will be given.
    spectrum : numpy.ndarray
        Spectrum to be smoothed.
    header : `~astropy.io.fits.Header`
        Header of the file containing the `spectrum`.
        Will be passed to :func:`~.aslsq_helper.check_signal_ranges`.
    check_signal_sigma : float, optional
        Defines the significance of the signal that has to be present in the spectra
        for at least the range defined by `velo_range`. Default is 6.0.
    noise : float
        Noise level of the data.
        Will be passed to :func:`~.aslsq_helper.check_signal_ranges`
    velo_range : float, optional
        Velocity range [in km/s] of the spectra that has to contain significant signal
        for it to be considered in the baseline extraction. Default is 15.0.
        Will be passed to :func:`~.aslsq_helper.check_signal_ranges`.
    niters : int, optional
        Maximum number of iterations of the smoothing.
        Default is 20.
    iterations_for_convergence : int, optional
        Number of iterations of the major cycle for the baseline to be considered converged.
        Default is 3.
    add_residual : bool, optional
        Whether to add the residual (=difference between first and last major cycle iteration) to the baseline.
        Default is False.
    thresh : float
        Convergence threshold.
        If residual falls below this threshold for `iterations_for_convergence` iterations, the algorithm terminates the smoothing.
    p_limit : float, optional
        The p-limit of the Markov chain to estimate signal ranges in the spectra.
        Default is 0.02.

    Returns
    -------
    bg : numpy.ndarray
        Background spectrum w/o self-absorption.
    hisa : numpy.ndarray
        Inverted self-absorption spectrum (i.e. expressed as equivalent emission).
    iterations : int
        Number of iterations until algorithm converged.
    flag_map : int
        Flag whether background did/did not converge or whether spectrum does/does not contain signal.
        If flag is 1, the were no issues in the fit. If 0, fit did not converge or did not contain signal.
    """
    flag_map = 1.
    if check_signal_ranges(spectrum, header, sigma=check_signal_sigma, noise=noise, velo_range=velo_range):
        # TODO
        max_consec_ch = get_max_consecutive_channels(len(spectrum), p_limit)
        consecutive_channels, ranges = determine_peaks(spectrum, peak='both', amp_threshold=None)
        mask_ranges = ranges[np.where(consecutive_channels>=max_consec_ch)]
        mask = mask_channels(len(spectrum), mask_ranges, pad_channels=3, remove_intervals=None)

        spectrum_prior = baseline_als_optimized(spectrum, lam1, p1, niter=3, mask=mask)
        spectrum_firstfit = spectrum_prior
        converge_logic = np.array([])
        for n in range(niters+1):
            spectrum_prior = baseline_als_optimized(spectrum_prior, lam1, p1, niter=3, mask=mask)
            spectrum_next = baseline_als_optimized(spectrum_prior, lam1, p1, niter=3, mask=mask)
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
                i_converge = np.min(np.argwhere(c > iterations_for_convergence)) + 2
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
        if np.all(mask):
            noise_fit_offset = thresh
        else:
            noise_range = np.invert(mask)
            noise_fit_offset = np.nanmean(final_spec[noise_range])
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


def two_step_extraction(lam1 : float, p1 : float, lam2 : float, p2 : float, spectrum : np.ndarray = None, header : fits.Header = None,
                        check_signal_sigma : float = 6., noise : float = None, velo_range : float = 15.0,
                        niters : int = 20, iterations_for_convergence : int = 3, add_residual : bool = False,
                        thresh : float = None, p_limit : float = 0.02) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Baseline smoothing routine using one lambda smoothing value for all major iterations.

    Parameters
    ----------
    lam1 : float
        Smoothing weight of the very first smoothing iteration. Adjusts the amount of smoothing.
    p1 : float
        Asymmetry weight of the very first smoothing iteration. Adjusts how much weight positive or negative signals (wrt the smoothed baseline) will be given.
    lam2 : float
        Smoothing weight of all remaining smoothing iterations. Adjusts the amount of smoothing.
    p2 : float
        Asymmetry weight of all remaining smoothing iterations. Adjusts how much weight positive or negative signals (wrt the smoothed baseline) will be given.
    spectrum : numpy.ndarray
        Spectrum to be smoothed.
    header : `~astropy.io.fits.Header`
        Header of the file containing the `spectrum`.
        Will be passed to :func:`~.aslsq_helper.check_signal_ranges`.
    check_signal_sigma : float, optional
        Defines the significance of the signal that has to be present in the spectra
        for at least the range defined by `velo_range`. Default is 6.0.
    noise : float
        Noise level of the data.
        Will be passed to :func:`~.aslsq_helper.check_signal_ranges`
    velo_range : float, optional
        Velocity range [in km/s] of the spectra that has to contain significant signal
        for it to be considered in the baseline extraction. Default is 15.0.
        Will be passed to :func:`~.aslsq_helper.check_signal_ranges`.
    niters : int, optional
        Maximum number of iterations of the smoothing.
        Default is 20.
    iterations_for_convergence : int, optional
        Number of iterations of the major cycle for the baseline to be considered converged.
        Default is 3.
    add_residual : bool, optional
        Whether to add the residual (=difference between first and last major cycle iteration) to the baseline.
        Default is False.
    thresh : float
        Convergence threshold.
        If residual falls below this threshold for `iterations_for_convergence` iterations, the algorithm terminates the smoothing.
    p_limit : float, optional
        The p-limit of the Markov chain to estimate signal ranges in the spectra.
        Default is 0.02.

    Returns
    -------
    bg : numpy.ndarray
        Background spectrum w/o self-absorption.
    hisa : numpy.ndarray
        Inverted self-absorption spectrum (i.e. expressed as equivalent emission).
    iterations : int
        Number of iterations until algorithm converged.
    flag_map : int
        Flag whether background did/did not converge or whether spectrum does/does not contain signal.
        If flag is 1, the were no issues in the fit. If 0, fit did not converge or did not contain signal.
    """
    flag_map = 1.
    if check_signal_ranges(spectrum, header, sigma=check_signal_sigma, noise=noise, velo_range=velo_range):
        # TODO
        max_consec_ch = get_max_consecutive_channels(len(spectrum), p_limit)
        consecutive_channels, ranges = determine_peaks(spectrum, peak='both', amp_threshold=None)
        mask_ranges = ranges[np.where(consecutive_channels>=max_consec_ch)]
        mask = mask_channels(len(spectrum), mask_ranges, pad_channels=3, remove_intervals=None)

        spectrum_prior = baseline_als_optimized(spectrum, lam1, p1, niter=3, mask=mask)
        spectrum_firstfit = spectrum_prior
        converge_logic = np.array([])
        for n in range(niters+1):
            spectrum_prior = baseline_als_optimized(spectrum_prior, lam2, p2, niter=3, mask=mask)
            spectrum_next = baseline_als_optimized(spectrum_prior, lam2, p2, niter=3, mask=mask)
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
                i_converge = np.min(np.argwhere(c > iterations_for_convergence)) + 2
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
        if np.all(mask):
            noise_fit_offset = thresh
        else:
            noise_range = np.invert(mask)
            noise_fit_offset = np.nanmean(final_spec[noise_range])
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
