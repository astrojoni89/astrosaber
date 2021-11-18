import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from .utils.aslsq_fit import baseline_als_optimized
from .utils.quality_checks import goodness_of_fit, get_max_consecutive_channels, determine_peaks, mask_channels
from tqdm import trange, tqdm


def init(data):
    global ilist
    ilist = np.arange(len(data))


def parallel_process(array, function, n_jobs=4, use_kwargs=False, front_num=3):
    """A parallel version of the map function with a progress bar.
    Credit: http://danshiebler.com/2016-09-14-parallel-progress-bar/
    Args:
        array (array-like): An array to iterate over.
        function (function): A python function to apply to the elements of array
        n_jobs (int, default=16): The number of cores to use
        use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
            keyword arguments to function
        front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
            Useful for catching bugs
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
        for f in tqdm(as_completed(futures), **kwargs):
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
    Args:
        array (array-like): An array to iterate over.
        function (function): A python function to apply to the elements of array
        n_jobs (int, default=16): The number of cores to use
        use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
            keyword arguments to function
        front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
            Useful for catching bugs
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


def func(use_ncpus=None, function=None):
    # Multiprocessing code
    ncpus = multiprocessing.cpu_count()
    # p = multiprocessing.Pool(ncpus, init_worker)
    if use_ncpus is None:
        use_ncpus = int(ncpus*0.75)
    print('\nUsing {} of {} cpus'.format(use_ncpus, ncpus))
    try:
        if function is None:
            raise ValueError('Have to set function for parallel process.')
        results_list = parallel_process(ilist, function=function, n_jobs=use_ncpus)
        
    except KeyboardInterrupt:
        print("KeyboardInterrupt... quitting.")
        quit()
    return results_list


def func_wo_bar(use_ncpus=None, function='cost'):
    # Multiprocessing code
    ncpus = multiprocessing.cpu_count()
    # p = multiprocessing.Pool(ncpus, init_worker)
    if use_ncpus is None:
        use_ncpus = int(ncpus*0.75)
    #print('Using {} of {} cpus'.format(use_ncpus, ncpus))
    try:
        if function is None:
            raise ValueError('Have to set function for parallel process.')
        results_list = parallel_process_wo_bar(ilist, function=function, n_jobs=use_ncpus)
    except KeyboardInterrupt:
        print("KeyboardInterrupt... quitting.")
        quit()
    return results_list
