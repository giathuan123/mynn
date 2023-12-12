from OptTensor import OptTensor
import numpy as np

t = OptTensor(np.random.randn(100, 500) )
x = np.random.randn(1000, 100)
print(t(x).shape)
