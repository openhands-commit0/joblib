"""
My own variation on function-specific inspect-like features.
"""
import inspect
import warnings
import re
import os
import collections
from itertools import islice
from tokenize import open as open_py_source
from .logger import pformat
full_argspec_fields = 'args varargs varkw defaults kwonlyargs kwonlydefaults annotations'
full_argspec_type = collections.namedtuple('FullArgSpec', full_argspec_fields)

def get_func_code(func):
    """ Attempts to retrieve a reliable function code hash.

        The reason we don't use inspect.getsource is that it caches the
        source, whereas we want this to be modified on the fly when the
        function is modified.

        Returns
        -------
        func_code: string
            The function code
        source_file: string
            The path to the file in which the function is defined.
        first_line: int
            The first line of the code in the source file.

        Notes
        ------
        This function does a bit more magic than inspect, and is thus
        more robust.
    """
    source_file = None
    try:
        source_file = inspect.getsourcefile(func)
    except:
        source_file = None

    if source_file is None:
        try:
            source_file = inspect.getfile(func)
        except:
            source_file = None

    if source_file is None:
        return None, None, None

    try:
        source_lines = inspect.findsource(func)
        source_lines, first_line = source_lines
        source_lines = ''.join(source_lines)
    except:
        return None, None, None

    return source_lines, source_file, first_line

def _clean_win_chars(string):
    """Windows cannot encode some characters in filename."""
    import urllib.parse
    if os.name == 'nt':
        return urllib.parse.quote(string, safe='')
    return string

def get_func_name(func, resolv_alias=True, win_characters=True):
    """ Return the function import path (as a list of module names), and
        a name for the function.

        Parameters
        ----------
        func: callable
            The func to inspect
        resolv_alias: boolean, optional
            If true, possible local aliases are indicated.
        win_characters: boolean, optional
            If true, substitute special characters using urllib.quote
            This is useful in Windows, as it cannot encode some filenames
    """
    if hasattr(func, '__module__'):
        module = func.__module__
    else:
        try:
            module = inspect.getmodule(func)
            if module is not None:
                module = module.__name__
        except:
            module = None
    if module is None:
        module = ''

    module_parts = module.split('.')

    if hasattr(func, '__name__'):
        name = func.__name__
    else:
        name = 'unknown'
        if hasattr(func, '__class__'):
            name = func.__class__.__name__

    if win_characters:
        name = _clean_win_chars(name)

    if resolv_alias:
        # Attempt to resolve name aliases using inspect
        if hasattr(func, '__code__'):
            try:
                code = func.__code__
                filename = code.co_filename
                first_line = code.co_firstlineno
                name = '%s-%d' % (name, first_line)
            except:
                pass

    return module_parts, name

def _signature_str(function_name, arg_sig):
    """Helper function to output a function signature"""
    args = []
    if arg_sig.args:
        args.extend(arg_sig.args)
    if arg_sig.varargs:
        args.append('*' + arg_sig.varargs)
    if arg_sig.varkw:
        args.append('**' + arg_sig.varkw)
    return '%s(%s)' % (function_name, ', '.join(args))

def _function_called_str(function_name, args, kwargs):
    """Helper function to output a function call"""
    parts = []
    if args:
        parts.extend(repr(arg) for arg in args)
    if kwargs:
        parts.extend('%s=%r' % (k, v) for k, v in sorted(kwargs.items()))
    return '%s(%s)' % (function_name, ', '.join(parts))

def filter_args(func, ignore_lst, args=(), kwargs=dict()):
    """ Filters the given args and kwargs using a list of arguments to
        ignore, and a function specification.

        Parameters
        ----------
        func: callable
            Function giving the argument specification
        ignore_lst: list of strings
            List of arguments to ignore (either a name of an argument
            in the function spec, or '*', or '**')
        *args: list
            Positional arguments passed to the function.
        **kwargs: dict
            Keyword arguments passed to the function

        Returns
        -------
        filtered_args: list
            List of filtered positional and keyword arguments.
    """
    arg_spec = inspect.getfullargspec(func)
    arg_names = list(arg_spec.args)
    output_args = list()

    # Filter positional arguments
    if '*' not in ignore_lst:
        for arg_name, arg in zip(arg_names, args):
            if arg_name not in ignore_lst:
                output_args.append(arg)

    # Filter keyword arguments
    if '**' not in ignore_lst:
        for arg_name in arg_names[len(args):]:
            if arg_name in kwargs:
                if arg_name not in ignore_lst:
                    output_args.append(kwargs[arg_name])
            else:
                # Check if the parameter has a default value
                default_arg = arg_spec.defaults[arg_names.index(arg_name) - len(arg_names)]
                if default_arg not in ignore_lst:
                    output_args.append(default_arg)

    return output_args

def format_call(func, args, kwargs, object_name='Memory'):
    """ Returns a nicely formatted statement displaying the function
        call with the given arguments.
    """
    path, name = get_func_name(func)
    path = [object_name] + list(path)
    module_path = '.'.join(path)

    arg_str = _function_called_str(name, args, kwargs)
    return '%s.%s' % (module_path, arg_str)

def format_signature(func):
    """Return a formatted signature for the function."""
    arg_spec = inspect.getfullargspec(func)
    path, name = get_func_name(func)
    module_path = '.'.join(path)

    signature = _signature_str(name, arg_spec)
    return '%s.%s' % (module_path, signature)