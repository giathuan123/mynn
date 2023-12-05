import torch as t  
import numpy as np


dim_in = 100
nn = t.randn(dim_in, requires_grad=True)
# Ground truth
gt_weights = t.randint(1, 10, size=(100,))

# Generate ground truth data
data_points = 20_000
data = t.randint(1, 10, size=(data_points, 100))
y = data@gt_weights
sgd = t.optim.SGD(nn, lr=0.01)
for d, label in zip(data, y):
    sgd.zero_grad()
    out = d@nn
    loss = (label - out)**2
    loss.backward()
    sgd.step()

nn.requires_grad_(False)
error = sum(nn - gt_weights)
print(error)
    
