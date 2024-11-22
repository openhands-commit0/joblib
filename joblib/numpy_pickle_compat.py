"""Numpy pickle compatibility functions."""
import pickle
import os
import zlib
import inspect
from io import BytesIO
from .numpy_pickle_utils import _ZFILE_PREFIX
from .numpy_pickle_utils import Unpickler
from .numpy_pickle_utils import _ensure_native_byte_order

def hex_str(an_int):
    """Convert an int to an hexadecimal string."""
    return hex(an_int)[2:]
_MAX_LEN = len(hex_str(2 ** 64))
_CHUNK_SIZE = 64 * 1024

def read_zfile(file_handle):
    """Read the z-file and return the content as a string.

    Z-files are raw data compressed with zlib used internally by joblib
    for persistence. Backward compatibility is not guaranteed. Do not
    use for external purposes.
    """
    file_handle.seek(0)
    header = file_handle.read(len(_ZFILE_PREFIX))
    if header != _ZFILE_PREFIX:
        raise ValueError("Unknown file type")

    length = file_handle.read(_MAX_LEN)
    length = int(length, 16)

    # Decompress small files in memory
    data = BytesIO()
    chunk = file_handle.read(_CHUNK_SIZE)
    decompressor = zlib.decompressobj()
    while chunk:
        data.write(decompressor.decompress(chunk))
        chunk = file_handle.read(_CHUNK_SIZE)
    data.write(decompressor.flush())
    data = data.getvalue()
    if len(data) != length:
        raise ValueError("File corrupted")
    return data

def write_zfile(file_handle, data, compress=1):
    """Write the data in the given file as a Z-file.

    Z-files are raw data compressed with zlib used internally by joblib
    for persistence. Backward compatibility is not guaranteed. Do not
    use for external purposes.
    """
    file_handle.write(_ZFILE_PREFIX)
    length = hex_str(len(data))
    # Add padding to length to make it fixed width
    file_handle.write(length.zfill(_MAX_LEN).encode('ascii'))
    compressor = zlib.compressobj(compress)
    chunk = data[0:_CHUNK_SIZE]
    pos = _CHUNK_SIZE
    while chunk:
        file_handle.write(compressor.compress(chunk))
        chunk = data[pos:pos + _CHUNK_SIZE]
        pos += _CHUNK_SIZE
    file_handle.write(compressor.flush())

class NDArrayWrapper(object):
    """An object to be persisted instead of numpy arrays.

    The only thing this object does, is to carry the filename in which
    the array has been persisted, and the array subclass.
    """

    def __init__(self, filename, subclass, allow_mmap=True):
        """Constructor. Store the useful information for later."""
        self.filename = filename
        self.subclass = subclass
        self.allow_mmap = allow_mmap

    def read(self, unpickler):
        """Reconstruct the array."""
        filename = os.path.join(unpickler._dirname, self.filename)
        # Load the array from the disk
        np = unpickler.np
        if np is None:
            raise ImportError("Trying to unpickle an ndarray, "
                            "but numpy is not available")
        array = _ensure_native_byte_order(np.load(filename, mmap_mode=unpickler.mmap_mode if self.allow_mmap else None))
        # Reconstruct subclasses. This does not work with old
        # versions of numpy
        if (not np.issubdtype(array.dtype, np.dtype('O')) and
                self.subclass not in (type(None), type(array))):
            new_array = np.ndarray.__new__(self.subclass, array.shape,
                                         array.dtype, buffer=array)
            # Preserve side effects of viewing arrays
            new_array.__array_finalize__(array)
            array = new_array
        return array

class ZNDArrayWrapper(NDArrayWrapper):
    """An object to be persisted instead of numpy arrays.

    This object store the Zfile filename in which
    the data array has been persisted, and the meta information to
    retrieve it.
    The reason that we store the raw buffer data of the array and
    the meta information, rather than array representation routine
    (tobytes) is that it enables us to use completely the strided
    model to avoid memory copies (a and a.T store as fast). In
    addition saving the heavy information separately can avoid
    creating large temporary buffers when unpickling data with
    large arrays.
    """

    def __init__(self, filename, init_args, state):
        """Constructor. Store the useful information for later."""
        self.filename = filename
        self.state = state
        self.init_args = init_args

    def read(self, unpickler):
        """Reconstruct the array from the meta-information and the z-file."""
        # Get the array parameters
        init_args, state = self.init_args, self.state

        # Read the array data from the z-file
        filename = os.path.join(unpickler._dirname, self.filename)
        with open(filename, 'rb') as f:
            array_bytes = read_zfile(f)

        # Reconstruct the array
        np = unpickler.np
        if np is None:
            raise ImportError("Trying to unpickle an ndarray, "
                            "but numpy is not available")
        array = np.ndarray(*init_args)
        array.__setstate__(state)
        array.data = np.frombuffer(array_bytes, dtype=array.dtype)
        return array

class ZipNumpyUnpickler(Unpickler):
    """A subclass of the Unpickler to unpickle our numpy pickles."""
    dispatch = Unpickler.dispatch.copy()

    def __init__(self, filename, file_handle, mmap_mode=None):
        """Constructor."""
        self._filename = os.path.basename(filename)
        self._dirname = os.path.dirname(filename)
        self.mmap_mode = mmap_mode
        self.file_handle = self._open_pickle(file_handle)
        Unpickler.__init__(self, self.file_handle)
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np

    def load_build(self):
        """Set the state of a newly created object.

        We capture it to replace our place-holder objects,
        NDArrayWrapper, by the array we are interested in. We
        replace them directly in the stack of pickler.
        """
        stack = self.stack
        state = stack.pop()
        instance = stack[-1]
        if isinstance(instance, NDArrayWrapper):
            # We replace the wrapper by the array
            array = instance.read(self)
            stack[-1] = array
            return
        setstate = getattr(instance, "__setstate__", None)
        if setstate is not None:
            setstate(state)
            return
        slotstate = None
        if isinstance(state, tuple) and len(state) == 2:
            state, slotstate = state
        if state:
            instance_dict = instance.__dict__
            for k, v in state.items():
                instance_dict[k] = v
        if slotstate:
            for k, v in slotstate.items():
                setattr(instance, k, v)
    dispatch[pickle.BUILD[0]] = load_build

def load_compatibility(filename):
    """Reconstruct a Python object from a file persisted with joblib.dump.

    This function ensures the compatibility with joblib old persistence format
    (<= 0.9.3).

    Parameters
    ----------
    filename: string
        The name of the file from which to load the object

    Returns
    -------
    result: any Python object
        The object stored in the file.

    See Also
    --------
    joblib.dump : function to save an object

    Notes
    -----

    This function can load numpy array files saved separately during the
    dump.
    """
    with open(filename, 'rb') as file_handle:
        # We are careful to open the file handle early and keep it open to
        # avoid race-conditions on renames.
        # XXX: This code should be refactored to use a context handler
        unpickler = ZipNumpyUnpickler(filename, file_handle)
        obj = unpickler.load()
    return obj