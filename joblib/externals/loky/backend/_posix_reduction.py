import os
import socket
import _socket
from multiprocessing.connection import Connection
from multiprocessing.context import get_spawning_popen
from .reduction import register
HAVE_SEND_HANDLE = hasattr(socket, 'CMSG_LEN') and hasattr(socket, 'SCM_RIGHTS') and hasattr(socket.socket, 'sendmsg')

def _mk_inheritable(fd):
    """Make a file descriptor inheritable by child processes."""
    os.set_inheritable(fd, True)
    return fd

def DupFd(fd):
    """Return a wrapper for an fd."""
    popen = get_spawning_popen()
    if popen is not None:
        return popen.DupFd(fd)
    else:
        return _mk_inheritable(os.dup(fd))

def _reduce_socket(s):
    """Reduce a socket object."""
    if HAVE_SEND_HANDLE and getattr(s, '_inheritable', False):
        return s, (None,)
    else:
        return _rebuild_socket, (DupFd(s.fileno()),
                               s.family, s.type, s.proto)

def _rebuild_socket(fd, family, type, proto):
    """Rebuild a socket object."""
    s = socket.socket(family, type, proto, fileno=fd)
    if HAVE_SEND_HANDLE:
        s._inheritable = True
    return s

def rebuild_connection(df, readable, writable):
    """Rebuild a connection object."""
    fd = df
    if not isinstance(fd, int):
        fd = fd.detach()
    conn = Connection(fd, readable, writable)
    conn._inheritable = True
    return conn

def reduce_connection(conn):
    """Reduce a connection object."""
    df = DupFd(conn.fileno())
    return rebuild_connection, (df, conn.readable, conn.writable)
register(socket.socket, _reduce_socket)
register(_socket.socket, _reduce_socket)
register(Connection, reduce_connection)