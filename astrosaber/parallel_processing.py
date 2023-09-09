import numpy as np
import multiprocessing
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from .training import saberTraining
from .prepare_training import saberPrepare
from .hisa import HisaExtraction
#from .utils.aslsq_fit import baseline_als_optimized
#from .utils.quality_checks import goodness_of_fit, get_max_consecutive_channels, determine_peaks, mask_channels
from tqdm import tqdm


    
def init(mp_info : List):
    '''
    Initializes global params for parallel process.
    
    '''
    global mp_ilist, mp_data, mp_params
    mp_data, mp_params = mp_info
    mp_ilist = np.arange(len(mp_data))
      
def single_cost_i(i : int) -> Tuple[float, float, float]:
    result = saberTraining.single_cost(mp_params[0], i)
    return result

def lambda_extraction_i(i : int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    result = saberPrepare.two_step_extraction_prepare(mp_params[0], i)
    return result

def two_step_i(i : int) -> Tuple[int, np.ndarray, np.ndarray, int, int]:
    result = HisaExtraction.two_step_extraction_single(mp_params[0], i)
    return result

def one_step_i(i : int) -> Tuple[int, np.ndarray, np.ndarray, int, int]:
    result = HisaExtraction.one_step_extraction_single(mp_params[0], i)
    return result


def parallel_process(array, function, n_jobs=4, use_kwargs=False, front_num=3, bar=tqdm):
    """
    A parallel version of the map function with a progress bar.
    Credit: http://danshiebler.com/2016-09-14-parallel-progress-bar/

    array : numpy.ndarray 
        An array to iterate over.
    function : func
        A python function to apply to the elements of array.
    n_jobs : int
        The number of cores to use.
    use_kwargs : bool
        Whether to consider the elements of array as dictionaries of
        keyword arguments to function. Default is False.
    front_num : int
        The number of iterations to run serially before kicking off the parallel job.
        Useful for catching bugs. Default is 3.
    Returns:
        [function(array[0]), function(array[1]), ...]
    """
    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]] #, lam1_updt=lam1_updt, p1_updt=p1_updt, lam2_updt=lam2_updt, p2_updt=p2_updt
    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    # Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]] # , lam1_updt=lam1_updt, p1_updt=p1_updt, lam2_updt=lam2_updt, p2_updt=p2_updt
        kwargs = {
            'total': len(futures),
            'unit': 'spec',
            'unit_scale': True,
            'leave': True
        }
        # Print out the progress as tasks complete
        for f in bar(as_completed(futures), **kwargs):
            pass
    out = []
    # Get the results from the futures.
    for i, future in enumerate(futures): #tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out


def parallel_process_wo_bar(array, function, n_jobs=4, use_kwargs=False, front_num=3):
    """A parallel version of the map function with a progress bar.
    Credit: http://danshiebler.com/2016-09-14-parallel-progress-bar/

    array : numpy.ndarray 
        An array to iterate over.
    function : func
        A python function to apply to the elements of array.
    n_jobs : int
        The number of cores to use.
    use_kwargs : bool
        Whether to consider the elements of array as dictionaries of
        keyword arguments to function. Default is False.
    front_num : int
        The number of iterations to run serially before kicking off the parallel job.
        Useful for catching bugs. Default is 3.
    Returns:
        [function(array[0]), function(array[1]), ...]
    """
    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]] #, lam1_updt=lam1_updt, p1_updt=p1_updt, lam2_updt=lam2_updt, p2_updt=p2_updt
    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    # Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]] # , lam1_updt=lam1_updt, p1_updt=p1_updt, lam2_updt=lam2_updt, p2_updt=p2_updt
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        # Print out the progress as tasks complete
        #for f in tqdm(as_completed(futures), **kwargs):
        #    pass
    out = []
    # Get the results from the futures.
    for i, future in enumerate(futures): #tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out


def func(use_ncpus=None, function=None, bar=tqdm):
    # Multiprocessing code
    ncpus = multiprocessing.cpu_count()
    # p = multiprocessing.Pool(ncpus, init_worker)
    if use_ncpus is None:
        use_ncpus = int(ncpus*0.75)
    print('\nUsing {} of {} cpus'.format(use_ncpus, ncpus))
    if mp_ilist is None:
        raise ValueError("Must specify 'mp_ilist'.")
    try:
        if function is None:
            raise ValueError('Have to set function for parallel process.')
        if function == 'two_step':
            results_list = parallel_process(mp_ilist, two_step_i, n_jobs=use_ncpus, bar=bar)
        if function == 'one_step':
            results_list = parallel_process(mp_ilist, one_step_i, n_jobs=use_ncpus, bar=bar)
        if function == 'cost':
            results_list = parallel_process(mp_ilist, single_cost_i, n_jobs=use_ncpus, bar=bar)
        if function == 'hisa':
            results_list = parallel_process(mp_ilist, lambda_extraction_i, n_jobs=use_ncpus, bar=bar)
        
    except KeyboardInterrupt:
        print("KeyboardInterrupt... quitting.")
        quit()
    return results_list


def func_wo_bar(use_ncpus=None, function=None):
    # Multiprocessing code
    ncpus = multiprocessing.cpu_count()
    # p = multiprocessing.Pool(ncpus, init_worker)
    if use_ncpus is None:
        use_ncpus = int(ncpus*0.75)
    #print('Using {} of {} cpus'.format(use_ncpus, ncpus))
    if mp_ilist is None:
        raise ValueError("Must specify 'mp_ilist'.")
    try:
        if function is None:
            raise ValueError('Have to set function for parallel process.')
        if function == 'two_step':
            results_list = parallel_process_wo_bar(mp_ilist, two_step_i, n_jobs=use_ncpus)
        if function == 'one_step':
            results_list = parallel_process_wo_bar(mp_ilist, one_step_i, n_jobs=use_ncpus)
        if function == 'cost':
            results_list = parallel_process_wo_bar(mp_ilist, single_cost_i, n_jobs=use_ncpus)
        if function == 'hisa':
            results_list = parallel_process_wo_bar(mp_ilist, lambda_extraction_i, n_jobs=use_ncpus)
    except KeyboardInterrupt:
        print("KeyboardInterrupt... quitting.")
        quit()
    return results_list
