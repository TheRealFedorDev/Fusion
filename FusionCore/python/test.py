import numpy as np

import sys
sys.path.insert(0, '.')
from fusion.tensor import tensor
from fusion.linear_model import LinearRegression

a = tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
b = tensor([0.5, 0.5, 0.5, 0.5], [2, 2])
c = a + b
print("Sum:", c.sum())
print("Shape:", c.shape)