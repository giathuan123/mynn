from OptTensor import OptTensor
import numpy as np


def header(heading: str) -> str:
    return '-'*5 + heading + '-'*5
t = OptTensor(np.random.randn(3, 2))
x = t.softmax()
x.backward()
print(header('input'))
print(t.data)
print(header('grad'))
edited = t.grad.copy()
edited[:, 1] = 0
print(t.grad)
print(edited)
print(header('output'))
print(x.data)
print(header('sum'))
print(x.data.sum(axis=1))
