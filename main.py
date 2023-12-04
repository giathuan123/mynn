from NeuralNetwork import MLP
import numpy as np
import random

dim_in = 3
dim_outs = [2, 1]
mlp = MLP(dim_in, dim_outs, non_linear_activation=True)
data = np.random.randint(1, 10, size=(20, 3))
# ground truth: y = x_2 (x_1 + x_0)
y = data[:, 2]*(data[:, 1] + data[:, 0])

# training loop
for xi, label in zip(data, y):
    mlp.zero_grad()
    out = mlp(xi)[0]
    loss = (y - out)**2
    loss.backward()
