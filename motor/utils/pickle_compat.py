"""
Loading the datasets' pickles across NumPy versions.

A pickle written under NumPy 2 names NumPy's internals 'numpy._core'; NumPy 1
calls the same module 'numpy.core' and cannot load it (ModuleNotFoundError:
No module named 'numpy._core'). load_pickle() maps one to the other when
needed, so data made on one machine loads on another (e.g. a laptop's NumPy 1
environment reading trials pickled under NumPy 2). NumPy 2 still loads
'numpy.core' pickles itself.
"""

import pickle

import numpy as np

_NUMPY_1 = int(np.__version__.split(".")[0]) < 2


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if _NUMPY_1 and (module == "numpy._core" or module.startswith("numpy._core.")):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def load_pickle(f):
    """pickle.load() of an open binary file, across NumPy 1 and 2."""
    return _CompatUnpickler(f).load()
