# python/fusion/tensor.py
import sys
sys.path.insert(0, r'C:\Users\Фёдор\CLionProjects\untitled4\FusionCore\cmake-build-debug')
from _fusion_core import Tensor as _Tensor

def tensor(data, shape=None):
    if shape is None:
        shape = [len(data)]
    return _Tensor(data, shape)

def zeros(shape):
    return _Tensor.zeros(shape)