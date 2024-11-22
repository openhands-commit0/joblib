"""
Backports of fixes for joblib dependencies
"""
import os
import re
import time
from os.path import basename
from multiprocessing import util

class Version:
    """Backport from deprecated distutils

    We maintain this backport to avoid introducing a new dependency on
    `packaging`.

    We might rexplore this choice in the future if all major Python projects
    introduce a dependency on packaging anyway.
    """

    def __init__(self, vstring=None):
        if vstring:
            self.parse(vstring)

    def __repr__(self):
        return "%s ('%s')" % (self.__class__.__name__, str(self))

    def __eq__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c == 0

    def __lt__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c < 0

    def __le__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c <= 0

    def __gt__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c > 0

    def __ge__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c >= 0

class LooseVersion(Version):
    """Backport from deprecated distutils

    We maintain this backport to avoid introducing a new dependency on
    `packaging`.

    We might rexplore this choice in the future if all major Python projects
    introduce a dependency on packaging anyway.
    """
    component_re = re.compile('(\\d+ | [a-z]+ | \\.)', re.VERBOSE)

    def __init__(self, vstring=None):
        if vstring:
            self.parse(vstring)

    def __str__(self):
        return self.vstring

    def __repr__(self):
        return "LooseVersion ('%s')" % str(self)
try:
    import numpy as np

    def make_memmap(filename, dtype='uint8', mode='r+', offset=0, shape=None, order='C', unlink_on_gc_collect=False):
        """Custom memmap constructor compatible with numpy.memmap.

        This function:
        - is a backport the numpy memmap offset fix (See
          https://github.com/numpy/numpy/pull/8443 for more details.
          The numpy fix is available starting numpy 1.13)
        - adds ``unlink_on_gc_collect``, which specifies  explicitly whether
          the process re-constructing the memmap owns a reference to the
          underlying file. If set to True, it adds a finalizer to the
          newly-created memmap that sends a maybe_unlink request for the
          memmaped file to resource_tracker.
        """
        if unlink_on_gc_collect:
            util.finalize(np.memmap(filename, dtype=dtype, mode=mode,
                                  offset=offset, shape=shape, order=order),
                        util.get_context().resource_tracker.maybe_unlink,
                        [filename])
        return np.memmap(filename, dtype=dtype, mode=mode,
                        offset=offset, shape=shape, order=order)
except ImportError:
    if os.name == 'nt':
        access_denied_errors = (5, 13)
        from os import replace

        def concurrency_safe_rename(src, dst):
            """Renames ``src`` into ``dst`` overwriting ``dst`` if it exists.

            On Windows os.replace can yield permission errors if executed by two
            different processes.
            """
            max_retries = 10
            retry_delay = 0.001  # Initial delay set to 1ms
            for i in range(max_retries):
                try:
                    replace(src, dst)
                    break
                except OSError as e:
                    if e.winerror not in access_denied_errors:
                        raise
                    if i == max_retries - 1:
                        raise
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
    else:
        from os import replace as concurrency_safe_rename