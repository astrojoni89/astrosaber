# THIS CODE IS TAKEN FROM THE BUNNY ADAPTATION OF TQDM
# check out https://github.com/bheinzerling/bunny

import sys
import multiprocessing as mp
import threading as th
from tqdm import tqdm
from tqdm import TqdmDeprecationWarning
from tqdm.utils import _term_move_up

up = _term_move_up()


try:
    mp_lock = mp.RLock()  # multiprocessing lock
except ImportError:  # pragma: no cover
    mp_lock = None
except OSError:  # pragma: no cover
    mp_lock = None
try:
    th_lock = th.RLock()  # thread lock
except OSError:  # pragma: no cover
    th_lock = None


class TqdmDefaultWriteLock(object):

    def __init__(self):
        global mp_lock, th_lock
        self.locks = [lk for lk in [mp_lock, th_lock] if lk is not None]

    def acquire(self):
        for lock in self.locks:
            lock.acquire()

    def release(self):
        for lock in self.locks[::-1]:
            lock.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, *exc):
        self.release()


class yoda(tqdm):
    monitor_interval = 10  # set to 0 to disable the thread
    monitor = None
    _lock = TqdmDefaultWriteLock()

    def __init__(self, iterable, **kwargs):
        super().__init__(iterable, **kwargs)

    def __iter__(self):
        """Backward-compatibility to use: for x in tqdm(iterable)"""

        iterable = self.iterable

        # If the bar is disabled, then just walk the iterable
        # (note: keep this check outside the loop for performance)
        if self.disable:
            for obj in iterable:
                yield obj
        else:
            mininterval = self.mininterval
            maxinterval = self.maxinterval
            miniters = self.miniters
            dynamic_miniters = self.dynamic_miniters
            last_print_t = self.last_print_t
            last_print_n = self.last_print_n
            n = self.n
            smoothing = self.smoothing
            #avg_time = self.avg_time
            _time = self._time

            try:
                sp = self.sp
            except AttributeError:
                raise TqdmDeprecationWarning("""\
Please use `tqdm_gui(...)` instead of `tqdm(..., gui=True)`
""", fp_write=getattr(self.fp, 'write', sys.stderr.write))

            tqdm.write("\r" + " " * self.ncols + "\n" * 9)  # make space for baby yoda

            for obj in iterable:
                yield obj
                # Update and possibly print the progressbar.
                # Note: does not call self.update(1) for speed optimisation.
                n += 1
                # check counter first to avoid calls to time()
                if n - last_print_n >= self.miniters:
                    miniters = self.miniters  # watch monitoring thread changes
                    delta_t = _time() - last_print_t
                    if delta_t >= mininterval:
                        cur_t = _time()
                        delta_it = n - last_print_n

                        self.n = n
                        with self._lock:
                            if self.pos:
                                self.moveto(abs(self.pos))
                            # Print bar update
                            sp(self.__repr__())
                            if self.pos:
                                self.moveto(-abs(self.pos))

                        # If no `miniters` was specified, adjust automatically
                        # to the max iteration rate seen so far between 2 prints
                        if dynamic_miniters:
                            if maxinterval and delta_t >= maxinterval:
                                # Adjust miniters to time interval by rule of 3
                                if mininterval:
                                    # Set miniters to correspond to mininterval
                                    miniters = delta_it * mininterval / delta_t
                                else:
                                    # Set miniters to correspond to maxinterval
                                    miniters = delta_it * maxinterval / delta_t
                            elif smoothing:
                                miniters = smoothing * delta_it * \
                                    (mininterval / delta_t
                                     if mininterval and delta_t else 1) + \
                                    (1 - smoothing) * miniters
                            else:
                                # Maximum nb of iterations between 2 prints
                                miniters = max(miniters, delta_it)

                        # Store old values for next call
                        self.n = self.last_print_n = last_print_n = n
                        self.last_print_t = last_print_t = cur_t
                        self.miniters = miniters

                tqdm.write(up * 10)  # move cursor up
                if self.total:
                    # move baby yoda
                    offset = " " * int(n / self.total * (self.ncols - 40))
                else:
                    offset = ""
                #frac = n / self.total
                #percentage = frac * 100
                #tqdm.write(offset + '      |￣￣￣￣￣|')
                #tqdm.write(offset + '      | TRAINING |') 
                #tqdm.write(offset + '      |     step    |')
                #tqdm.write(offset + f'      |  {percentage:>4.0f}%   |')  
                #tqdm.write(offset + '      |＿＿＿＿＿|')
                tqdm.write(offset + '     ````')                            
                tqdm.write(offset + '   `     `')              
                tqdm.write(offset + ' ``     ``.````````') 
                tqdm.write(offset + '`       ``        `') 
                tqdm.write(offset + ' `               `')  
                tqdm.write(offset + '    `````  `````')               
                tqdm.write(offset + '                              ﹏    ') 
                tqdm.write(offset + '                          \033[92m<´(\033[0m⬬ ⬬\033[92m)`> ')
                tqdm.write(offset + '                           \033[92mʿ\033[0m/   \\\033[92mʾ\033[0m  ')

            # Closing the progress bar.
            # Update some internal variables for close().
            self.last_print_n = last_print_n
            self.n = n
            self.miniters = miniters
            self.close()
