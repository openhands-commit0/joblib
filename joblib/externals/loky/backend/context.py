import os
import sys
import math
import subprocess
import traceback
import warnings
import multiprocessing as mp
from multiprocessing import get_context as mp_get_context
from multiprocessing.context import BaseContext
from .process import LokyProcess, LokyInitMainProcess
if sys.version_info >= (3, 8):
    from concurrent.futures.process import _MAX_WINDOWS_WORKERS
    if sys.version_info < (3, 10):
        _MAX_WINDOWS_WORKERS = _MAX_WINDOWS_WORKERS - 1
else:
    _MAX_WINDOWS_WORKERS = 60
START_METHODS = ['loky', 'loky_init_main', 'spawn']
if sys.platform != 'win32':
    START_METHODS += ['fork', 'forkserver']
_DEFAULT_START_METHOD = None
physical_cores_cache = None

def cpu_count(only_physical_cores=False):
    """Return the number of CPUs the current process can use.

    The returned number of CPUs accounts for:
     * the number of CPUs in the system, as given by
       ``multiprocessing.cpu_count``;
     * the CPU affinity settings of the current process
       (available on some Unix systems);
     * Cgroup CPU bandwidth limit (available on Linux only, typically
       set by docker and similar container orchestration systems);
     * the value of the LOKY_MAX_CPU_COUNT environment variable if defined.
    and is given as the minimum of these constraints.

    If ``only_physical_cores`` is True, return the number of physical cores
    instead of the number of logical cores (hyperthreading / SMT). Note that
    this option is not enforced if the number of usable cores is controlled in
    any other way such as: process affinity, Cgroup restricted CPU bandwidth
    or the LOKY_MAX_CPU_COUNT environment variable. If the number of physical
    cores is not found, return the number of logical cores.

    Note that on Windows, the returned number of CPUs cannot exceed 61 (or 60 for
    Python < 3.10), see:
    https://bugs.python.org/issue26903.

    It is also always larger or equal to 1.
    """
    # Get the number of logical cores
    try:
        os_cpu_count = mp.cpu_count()
    except NotImplementedError:
        os_cpu_count = 1

    if sys.platform == 'win32':
        os_cpu_count = min(os_cpu_count, _MAX_WINDOWS_WORKERS)

    cpu_count_user = _cpu_count_user(os_cpu_count)
    if cpu_count_user is not None:
        return cpu_count_user

    if only_physical_cores:
        physical_cores, exception = _count_physical_cores()
        if physical_cores != "not found":
            return max(1, physical_cores)

    return max(1, os_cpu_count)

def _cpu_count_user(os_cpu_count):
    """Number of user defined available CPUs"""
    cpu_count_user = os.environ.get('LOKY_MAX_CPU_COUNT', None)
    if cpu_count_user is not None:
        if cpu_count_user.strip() == '':
            return None
        try:
            cpu_count_user = float(cpu_count_user)
            if cpu_count_user > 0:
                return int(min(cpu_count_user, os_cpu_count))
            else:
                return max(1, int(cpu_count_user * os_cpu_count))
        except ValueError:
            warnings.warn("LOKY_MAX_CPU_COUNT should be an integer or a float."
                        " Got '{}'. Using {} CPUs."
                        .format(cpu_count_user, os_cpu_count))
    return None

def _count_physical_cores():
    """Return a tuple (number of physical cores, exception)

    If the number of physical cores is found, exception is set to None.
    If it has not been found, return ("not found", exception).

    The number of physical cores is cached to avoid repeating subprocess calls.
    """
    global physical_cores_cache
    if physical_cores_cache is not None:
        return physical_cores_cache

    if sys.platform == 'linux':
        try:
            # Try to get the number of physical cores from /proc/cpuinfo
            with open('/proc/cpuinfo', 'rb') as f:
                cpuinfo = f.read().decode('ascii')
            cores = set()
            for line in cpuinfo.split('\n'):
                if line.startswith('physical id'):
                    phys_id = line.split(':')[1].strip()
                elif line.startswith('cpu cores'):
                    nb_cores = int(line.split(':')[1].strip())
                    cores.add((phys_id, nb_cores))
            if cores:
                physical_cores_cache = (sum(nb_cores for _, nb_cores in cores), None)
                return physical_cores_cache
        except Exception as e:
            physical_cores_cache = ("not found", e)
            return physical_cores_cache

    elif sys.platform == 'win32':
        try:
            # Try to get the number of physical cores from wmic
            cmd = ['wmic', 'cpu', 'get', 'NumberOfCores']
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = p.communicate()
            if p.returncode == 0:
                stdout = stdout.decode('ascii')
                cores = [int(l) for l in stdout.split('\n')[1:] if l.strip()]
                if cores:
                    physical_cores_cache = (sum(cores), None)
                    return physical_cores_cache
        except Exception as e:
            physical_cores_cache = ("not found", e)
            return physical_cores_cache

    elif sys.platform == 'darwin':
        try:
            # Try to get the number of physical cores from sysctl
            cmd = ['sysctl', '-n', 'hw.physicalcpu']
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = p.communicate()
            if p.returncode == 0:
                physical_cores_cache = (int(stdout.strip()), None)
                return physical_cores_cache
        except Exception as e:
            physical_cores_cache = ("not found", e)
            return physical_cores_cache

    physical_cores_cache = ("not found", "unknown platform")
    return physical_cores_cache

class LokyContext(BaseContext):
    """Context relying on the LokyProcess."""
    _name = 'loky'
    Process = LokyProcess
    cpu_count = staticmethod(cpu_count)

    def Queue(self, maxsize=0, reducers=None):
        """Returns a queue object"""
        from .queues import Queue
        return Queue(maxsize, reducers=reducers, ctx=self.get_context())

    def SimpleQueue(self, reducers=None):
        """Returns a queue object"""
        from .queues import SimpleQueue
        return SimpleQueue(reducers=reducers, ctx=self.get_context())
    if sys.platform != 'win32':
        'For Unix platform, use our custom implementation of synchronize\n        ensuring that we use the loky.backend.resource_tracker to clean-up\n        the semaphores in case of a worker crash.\n        '

        def Semaphore(self, value=1):
            """Returns a semaphore object"""
            from .synchronize import Semaphore
            return Semaphore(value, ctx=self.get_context())

        def BoundedSemaphore(self, value):
            """Returns a bounded semaphore object"""
            from .synchronize import BoundedSemaphore
            return BoundedSemaphore(value, ctx=self.get_context())

        def Lock(self):
            """Returns a lock object"""
            from .synchronize import Lock
            return Lock(ctx=self.get_context())

        def RLock(self):
            """Returns a recurrent lock object"""
            from .synchronize import RLock
            return RLock(ctx=self.get_context())

        def Condition(self, lock=None):
            """Returns a condition object"""
            from .synchronize import Condition
            return Condition(lock, ctx=self.get_context())

        def Event(self):
            """Returns an event object"""
            from .synchronize import Event
            return Event(ctx=self.get_context())

class LokyInitMainContext(LokyContext):
    """Extra context with LokyProcess, which does load the main module

    This context is used for compatibility in the case ``cloudpickle`` is not
    present on the running system. This permits to load functions defined in
    the ``main`` module, using proper safeguards. The declaration of the
    ``executor`` should be protected by ``if __name__ == "__main__":`` and the
    functions and variable used from main should be out of this block.

    This mimics the default behavior of multiprocessing under Windows and the
    behavior of the ``spawn`` start method on a posix system.
    For more details, see the end of the following section of python doc
    https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming
    """
    _name = 'loky_init_main'
    Process = LokyInitMainProcess
def get_context(method=None):
    """Returns a BaseContext or instance of subclass of BaseContext.
    
    method parameter can be 'fork', 'spawn', 'forkserver', 'loky' or None.
    If None, the default context is returned.
    """
    if method is None:
        # Get the default context
        if _DEFAULT_START_METHOD is None:
            _DEFAULT_START_METHOD = 'loky'
        method = _DEFAULT_START_METHOD
    
    if method not in START_METHODS:
        raise ValueError(
            "Method '{}' not in available methods {}".format(
                method, START_METHODS))
    
    if method == 'loky':
        return ctx_loky
    elif method == 'loky_init_main':
        return mp.context._concrete_contexts['loky_init_main']
    else:
        return mp_get_context(method)

ctx_loky = LokyContext()
mp.context._concrete_contexts['loky'] = ctx_loky
mp.context._concrete_contexts['loky_init_main'] = LokyInitMainContext()