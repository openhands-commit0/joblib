"""
Fast cryptographic hash of Python objects, with a special case for fast
hashing of numpy arrays.
"""
import pickle
import hashlib
import sys
import types
import struct
import io
import decimal
Pickler = pickle._Pickler

class _ConsistentSet(object):
    """ Class used to ensure the hash of Sets is preserved
        whatever the order of its items.
    """

    def __init__(self, set_sequence):
        try:
            self._sequence = sorted(set_sequence)
        except (TypeError, decimal.InvalidOperation):
            self._sequence = sorted((hash(e) for e in set_sequence))

class _MyHash(object):
    """ Class used to hash objects that won't normally pickle """

    def __init__(self, *args):
        self.args = args

class Hasher(Pickler):
    """ A subclass of pickler, to do cryptographic hashing, rather than
        pickling.
    """

    def __init__(self, hash_name='md5'):
        self.stream = io.BytesIO()
        protocol = 3
        Pickler.__init__(self, self.stream, protocol=protocol)
        self._hash = hashlib.new(hash_name)

    def save_global(self, obj, name=None, pack=struct.pack):
        """Save a global object"""
        self._hash.update(str(obj).encode('utf-8'))

    def save_set(self, obj, pack=struct.pack):
        """Save a set object"""
        self._hash.update(str(_ConsistentSet(obj)).encode('utf-8'))

    dispatch = Pickler.dispatch.copy()
    dispatch[type(len)] = save_global
    dispatch[type(object)] = save_global
    dispatch[type(Pickler)] = save_global
    dispatch[type(pickle.dump)] = save_global
    dispatch[type(set())] = save_set

class NumpyHasher(Hasher):
    """ Special case the hasher for when numpy is loaded.
    """

    def __init__(self, hash_name='md5', coerce_mmap=False):
        """
            Parameters
            ----------
            hash_name: string
                The hash algorithm to be used
            coerce_mmap: boolean
                Make no difference between np.memmap and np.ndarray
                objects.
        """
        self.coerce_mmap = coerce_mmap
        Hasher.__init__(self, hash_name=hash_name)
        import numpy as np
        self.np = np
        if hasattr(np, 'getbuffer'):
            self._getbuffer = np.getbuffer
        else:
            self._getbuffer = memoryview

    def save(self, obj):
        """ Subclass the save method, to hash ndarray subclass, rather
            than pickling them. Off course, this is a total abuse of
            the Pickler class.
        """
        if isinstance(obj, type):
            return Hasher.save_global(self, obj)
        if isinstance(obj, self.np.ndarray) and not obj.dtype.hasobject:
            # Compute a hash of the object
            try:
                self._hash.update(self._getbuffer(obj))
            except (TypeError, BufferError):
                # Cater for non-single-segment arrays: this creates a
                # copy, and thus aleviates this issue.
                # XXX: There might be a more efficient way of doing this
                self._hash.update(self._getbuffer(obj.flatten()))

            # We also hash the dtype and the shape to distinguish
            # different views of the same data with different dtypes.
            self._hash.update(str(obj.dtype).encode('utf-8'))
            self._hash.update(str(obj.shape).encode('utf-8'))
            return
        return Hasher.save(self, obj)

def hash(obj, hash_name='md5', coerce_mmap=False):
    """ Quick calculation of a hash to identify uniquely Python objects
        containing numpy arrays.

        Parameters
        ----------
        hash_name: 'md5' or 'sha1'
            Hashing algorithm used. sha1 is supposedly safer, but md5 is
            faster.
        coerce_mmap: boolean
            Make no difference between np.memmap and np.ndarray
    """
    try:
        import numpy as np
        hasher = NumpyHasher(hash_name=hash_name, coerce_mmap=coerce_mmap)
    except ImportError:
        hasher = Hasher(hash_name=hash_name)
    hasher.save(obj)
    return hasher._hash.hexdigest()