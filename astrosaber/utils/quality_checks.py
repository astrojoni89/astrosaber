# this code is taken from Lindner (2014) & Riener (2019); GaussPy(+) as is

import numpy as np

def goodness_of_fit(data, fit, errors, dof, mask=None,
                    get_aicc=False):
    """Determine the goodness of fit (reduced chi-square, AICc).
    
    Parameters
    ----------
    data : numpy.ndarray
        Original data.
    fit : numpy.ndarray
        Fit to the original data.
    errors : numpy.ndarray or float
        Root-mean-square noise for each channel.
    dof : int
        Degrees of freedom.
    mask : numpy.ndarray
        Boolean array specifying which regions of the spectrum should be used.
    get_aicc : bool
        If set to `True`, the AICc value will be returned in addition to the
        reduced chi2 value.

    Returns
    -------
    rchi2 : float
        Reduced chi2 value.
    aicc : float
        (optional): The AICc value is returned if get_aicc is set to `True`.
    """
    if type(errors) is not np.ndarray:
        errors = np.ones(len(data)) * errors
    # TODO: check if mask is set to None everywehere there is no mask
    if mask is None:
        mask = np.ones(len(data))
        mask = mask.astype('bool')
    elif len(mask) == 0:
        mask = np.ones(len(data))
        mask = mask.astype('bool')
    elif np.count_nonzero(mask) == 0:
        mask = np.ones(len(data))
        mask = mask.astype('bool')

    squared_residuals = (data[mask] - fit[mask])**2
    chi2 = np.nansum(squared_residuals / errors[mask]**2)
    n_samples = len(data[mask])
    rchi2 = chi2 / (n_samples - dof)
    if get_aicc:
        #  sum of squared residuals
        ssr = np.nansum(squared_residuals)
        log_likelihood = -0.5 * n_samples * np.log(ssr / n_samples)
        aicc = (2.0 * (dof - log_likelihood) +
                2.0 * dof * (dof + 1.0) /
                (n_samples - dof - 1.0))
        return rchi2, aicc
    return rchi2


def get_max_consecutive_channels(n_channels, p_limit):
    """Determine the maximum number of random consecutive positive/negative channels.
    Calculate the number of consecutive positive or negative channels,
    whose probability of occurring due to random chance in a spectrum
    is less than p_limit.

    Parameters
    ----------
    n_channels : int
        Number of spectral channels.
    p_limit : float
        Maximum probability for consecutive positive/negative channels being
        due to chance.

    Returns
    -------
    consec_channels : int
        Number of consecutive positive/negative channels that have a probability
        less than p_limit to be due to chance.
    """
    for consec_channels in range(2, 30):
        a = np.zeros((consec_channels, consec_channels))
        for i in range(consec_channels - 1):
            a[i, 0] = a[i, i + 1] = 0.5
        a[consec_channels - 1, consec_channels - 1] = 1.0
        if np.linalg.matrix_power(a, n_channels - 1)[0, consec_channels - 1] < p_limit:
            return consec_channels


def determine_peaks(spectrum, peak='both', amp_threshold=None):
    """Find peaks in a spectrum.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Array of the data values of the spectrum.
    peak : 'both' (default), 'positive', 'negative'
        Description of parameter `peak`.
    amp_threshold : float
        Required minimum threshold that at least one data point in a peak feature has to exceed.

    Returns
    -------
    consecutive_channels or amp_vals : numpy.ndarray
        If the 'amp_threshold' value is supplied an array with the maximum data values of the ranges is returned. Otherwise, the number of spectral channels of the ranges is returned.
    ranges : list
        List of intervals [(low, upp), ...] determined to contain peaks.
    """
    if (peak == 'both') or (peak == 'positive'):
        clipped_spectrum = spectrum.clip(max=0)
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
        iszero = np.concatenate(([0], np.equal(clipped_spectrum, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

    if (peak == 'both') or (peak == 'negative'):
        clipped_spectrum = spectrum.clip(min=0)
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
        iszero = np.concatenate(([0], np.equal(clipped_spectrum, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        if peak == 'both':
            # Runs start and end where absdiff is 1.
            ranges = np.append(ranges, np.where(absdiff == 1)[0].reshape(-1, 2), axis=0)
        else:
            ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

    if amp_threshold is not None:
        if peak == 'positive':
            mask = spectrum > abs(amp_threshold)
        elif peak == 'negative':
            mask = spectrum < -abs(amp_threshold)
        else:
            mask = np.abs(spectrum) > abs(amp_threshold)

        if np.count_nonzero(mask) == 0:
            return np.array([]), np.array([])

        peak_mask = np.split(mask, ranges[:, 1])
        mask_true = np.array([any(array) for array in peak_mask[:-1]])

        ranges = ranges[mask_true]
        if peak == 'positive':
            amp_vals = np.array([max(spectrum[low:upp]) for low, upp in ranges])
        elif peak == 'negative':
            amp_vals = np.array([min(spectrum[low:upp]) for low, upp in ranges])
        else:
            amp_vals = np.array([np.sign(spectrum[low])*max(np.abs(spectrum[low:upp])) for low, upp in ranges])
        #  TODO: check if sorting really necessary??
        sort_indices = np.argsort(amp_vals)[::-1]
        return amp_vals[sort_indices], ranges[sort_indices]
    else:
        sort_indices = np.argsort(ranges[:, 0])
        ranges = ranges[sort_indices]

        consecutive_channels = ranges[:, 1] - ranges[:, 0]
        return consecutive_channels, ranges


def mask_channels(n_channels, ranges, pad_channels=None, remove_intervals=None):
    """Determine the 1D boolean mask for a given list of spectral ranges.

    Parameters
    ----------
    n_channels : int
        Number of spectral channels.
    ranges : list
        List of intervals [(low, upp), ...].
    pad_channels : int
        Number of channels by which an interval (low, upp) gets extended on both sides, resulting in (low - pad_channels, upp + pad_channels).
    remove_intervals : type
        Nested list containing info about ranges of the spectrum that should be masked out.

    Returns
    -------
    mask : numpy.ndarray
        1D boolean mask that has 'True' values at the position of the channels contained in ranges.
    """
    mask = np.zeros(n_channels)

    for (lower, upper) in ranges:
        if pad_channels is not None:
            lower = max(0, lower - pad_channels)
            upper = min(n_channels, upper + pad_channels)
        mask[lower:upper] = 1

    if remove_intervals is not None:
        for (low, upp) in remove_intervals:
            mask[low:upp] = 0

    return mask.astype('bool')
