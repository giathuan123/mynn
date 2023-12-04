from NeuralNetwork import MLP
from matplotlib import pyplot as plt
import numpy as np

dim_in = 3
dim_outs = [1]
mlp = MLP(dim_in, dim_outs)
data_points = 20_000
data = np.random.randint(1, 10, size=(data_points, dim_in))
gt_weights = np.random.randint(1, 10, size=dim_in)
gt_bias = 10

y = np.matmul(data, gt_weights) + 10
data = data.tolist()
y = y.tolist()
EPOCHS = 20
out = None
learning_rate = 0.001
plt_loss = []
# training loop
for epoch in range(EPOCHS):
    ave_loss = 0
    for xi, value in zip(data, y):
        mlp.zero_grad()
        out = mlp(xi)[0]
        loss = (value - out)**2
        ave_loss += loss.data
        loss.backward()
        for p in mlp.params():
            p.data -= learning_rate * p.grad
    ave_loss /= data_points
    print(f'{epoch}:{ave_loss}')
    plt_loss.append(ave_loss)
    if ave_loss < 0.00001:
        break
plt.plot(np.arange(0, epoch+1), plt_loss)
weights = [v.data for v in mlp.layers[0].neurons[0].w]
bias = mlp.layers[0].neurons[0].b.data
print('weights: ', weights, 'bias: ', bias)
print('gt_weights: ', gt_weights, 'bias: ', gt_bias)
plt.show()
