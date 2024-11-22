import copyreg
import io
import functools
import types
import sys
import os
from multiprocessing import util
from pickle import loads, HIGHEST_PROTOCOL
from multiprocessing.reduction import register

_dispatch_table = {}

def _reduce_method(m):
    """Helper function for pickling methods."""
    if m.__self__ is None:
        return getattr, (m.__self__.__class__, m.__func__.__name__)
    else:
        return getattr, (m.__self__, m.__func__.__name__)

def _reduce_method_descriptor(m):
    """Helper function for pickling method descriptors."""
    return getattr, (m.__objclass__, m.__name__)

def _reduce_partial(p):
    """Helper function for pickling partial functions."""
    return _rebuild_partial, (p.func, p.args, p.keywords or {})

def _rebuild_partial(func, args, keywords):
    """Helper function for rebuilding partial functions."""
    return functools.partial(func, *args, **keywords)

class _C:
    def f(self):
        pass
    @classmethod
    def h(cls):
        pass
register(type(_C().f), _reduce_method)
register(type(_C.h), _reduce_method)
if not hasattr(sys, 'pypy_version_info'):
    register(type(list.append), _reduce_method_descriptor)
    register(type(int.__add__), _reduce_method_descriptor)
register(functools.partial, _reduce_partial)
if sys.platform != 'win32':
    from ._posix_reduction import _mk_inheritable
else:
    from . import _win_reduction
try:
    from joblib.externals import cloudpickle
    DEFAULT_ENV = 'cloudpickle'
except ImportError:
    DEFAULT_ENV = 'pickle'
ENV_LOKY_PICKLER = os.environ.get('LOKY_PICKLER', DEFAULT_ENV)
_LokyPickler = None
_loky_pickler_name = None
set_loky_pickler()

def set_loky_pickler(loky_pickler=None):
    """Select the pickler to use in loky.

    Parameters
    ----------
    loky_pickler: str in {'pickle', 'cloudpickle', None}, default=None
        If None, use the value of the environment variable LOKY_PICKLER.
        If 'pickle', use the standard pickle module.
        If 'cloudpickle', use the cloudpickle module.
    """
    global _LokyPickler, _loky_pickler_name

    if loky_pickler is None:
        loky_pickler = ENV_LOKY_PICKLER

    if loky_pickler == _loky_pickler_name:
        return

    if loky_pickler == 'pickle':
        from pickle import Pickler
        _LokyPickler = Pickler
    elif loky_pickler == 'cloudpickle':
        from joblib.externals.cloudpickle import CloudPickler
        _LokyPickler = CloudPickler
    else:
        raise ValueError(
            "Invalid value for LOKY_PICKLER: '{}'. Supported values are "
            "'pickle' and 'cloudpickle'".format(loky_pickler))
    _loky_pickler_name = loky_pickler

def dump(obj, file, reducers=None, protocol=None):
    """Replacement for pickle.dump() using _LokyPickler."""
    if protocol is None:
        protocol = HIGHEST_PROTOCOL
    _LokyPickler(file, protocol=protocol).dump(obj)

def dumps(obj, reducers=None, protocol=None):
    """Replacement for pickle.dumps() using _LokyPickler."""
    buf = io.BytesIO()
    dump(obj, buf, reducers=reducers, protocol=protocol)
    return buf.getbuffer()
__all__ = ['dump', 'dumps', 'loads', 'register', 'set_loky_pickler']
if sys.platform == 'win32':
    from multiprocessing.reduction import duplicate
    __all__ += ['duplicate']