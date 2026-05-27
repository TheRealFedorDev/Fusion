import numpy as np

import sys
sys.path.insert(0, 'fusion')
from fusion.tensor import Tensor

from fusion.linear_model import LinearRegression

from fusion.tensor import Tensor
X = Tensor([[1., 2., 3.], [4., 5., 6.]])
mean = X.mean(axis=0)          # тензор [2.5, 3.5, 4.5]
diff = X - mean