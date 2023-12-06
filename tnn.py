import torch as t  
import numpy as np


dim_in = 10
nn = t.nn.Linear(dim_in, 1, dtype=t.float)
# Ground truth
gt_weights = t.randint(1, 10, size=(dim_in,)).type(t.float)
# Generate ground truth data
data_points = 20_000
gt_bias = 10
data = t.randint(1, 10, size=(data_points, dim_in)).type(t.float)
y = data@gt_weights + gt_bias

EPOCHS = 20
sgd = t.optim.SGD(nn.parameters(), lr=0.0001)
for epoch in range(EPOCHS):
    ave_loss = 0
    for d, label in zip(data, y):
        nn.zero_grad()
        out = nn(d.type(t.float))
        loss = (label - out)**2
        ave_loss += loss.item()
        loss.backward()
        sgd.step()
    ave_loss /= data_points
    print(f'{epoch}:{ave_loss}')

nn.requires_grad_(False)
errors = ((nn.weight - gt_weights)**2).sum()
print("Errors: ", errors.item())
