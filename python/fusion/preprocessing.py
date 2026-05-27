# python/fusion/preprocessing.py

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from _fusion_core import Tensor as _Tensor
from .tensor import Tensor

def StandartScaler(X):
    m, n = X.shape[0], X.shape[1]
    mean = X.mean(axis=0)                     # shape (n,)
    # размножаем mean до (m, n)   (да-да размножаем)
    mean_2d = Tensor(mean.tolist() * m, [m, n])  # повторяется список m раз
    diff = X - mean_2d                         # теперь размеры совпадают, good
    var = (diff * diff).mean(axis=0)           # shape (n,)
    std = var ** 0.5                           # shape (n,)
    std_2d = Tensor(std.tolist() * m, [m, n])
    return (X - mean_2d) / std_2d