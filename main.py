from NeuralNetwork import MLP
from matplotlib import pyplot as plt
import numpy as np

dim_in = 3
dim_outs = [1]
mlp = MLP(dim_in, dim_outs, non_linear_activation=True)
data = np.random.randint(1, 10, size=(20, 3))
gt_weights: list[float] = [20, 1, 3]
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
    ave_loss /= 2000
    plt_loss.append(ave_loss) 
plt.plot(np.arange(1, EPOCHS + 1), plt_loss) 
weights = [v.data for v in mlp.layers[0].neurons[0].w]
bias = mlp.layers[0].neurons[0].b.data
print('weights: ', weights, 'bias: ', bias)
print('gt_weights: ', gt_weights, 'bias: ', gt_bias)
weights += [bias]
gt_weights += [gt_bias]
errors = sum((np.array(weights) - np.array(gt_weights))**2)
print(errors)
if out:
    graph = out.plot()
    graph.view()
plt.show()
