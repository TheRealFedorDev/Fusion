import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from _fusion_core import Tensor as _Tensor

def tensor(data, shape=None):
    if shape is None:
        shape = [len(data)]
    return _Tensor(data, shape)

def zeros(shape):
    return _Tensor.zeros(shape)