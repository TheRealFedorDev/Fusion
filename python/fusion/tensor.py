import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from _fusion_core import Tensor as _Tensor



def _flatten(nested):
    """ Рекурсивно превращает вложенные списки в плоский list"""
    if isinstance(nested, (list, tuple)):
        result = []
        for item in nested:
            result.extend(_flatten(item))
        return result
    else:
        return [float(nested)]

def _compute_shape(nested):
    """ Вычесление формы многомерного списка (все строки одинаковой длины)"""
    if isinstance(nested, (list, tuple)) and len(nested):
        if isinstance(nested[0], (list, tuple)):
            rest_shape = _compute_shape(nested[0])
            return [len(nested)] + rest_shape
        else:
            return [len(nested)]
    else:
        return [0] if len(nested) == 0 else [1]

def Tensor(data, shape=None):
    if shape is None:
        shape = _compute_shape(data)
    flat_data = _flatten(data)
    return _Tensor(flat_data, shape)

def zeros(shape):
    return _Tensor.zeros(shape)