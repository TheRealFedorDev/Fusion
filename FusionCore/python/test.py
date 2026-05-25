import numpy as np

import sys
sys.path.insert(0, 'fusion')
from fusion.tensor import Tensor

from fusion.linear_model import LinearRegression

a = Tensor([1.0, 2.0, 3.0], [3])
print(a)
print(a.tolist())