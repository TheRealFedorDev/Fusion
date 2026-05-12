import sys; sys.path.insert(0, r'C:\Users\Фёдор\CLionProjects\untitled4\FusionCore\cmake-build-debug')

from _fusion_core import Tensor

a = Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
b = Tensor([0.5, 0.5, 0.5, 0.5], [2, 2])

c = a + b
print("Сумма элементов c:", c.sum())         # Ожидается 12.0
print("Форма c:", c.shape)                   # [2, 2]

d = a * 2.0
print("Сумма d:", d.sum())                   # 20.0


# & "C:\Users\Фёдор\.conda\envs\untitled4\python.exe" "C:\Users\Фёдор\CLionProjects\untitled4\FusionCore\python\test.py"