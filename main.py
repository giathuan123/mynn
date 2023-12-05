from NeuralNetwork import MLP
from matplotlib import pyplot as plt
import numpy as np

<<<<<<< Updated upstream
dim_in = 100
dim_outs = [1]
data_points = 80000
mlp = MLP(dim_in, dim_outs)
data = np.random.randint(1, 10, size=(data_points, dim_in))
# ground truth weights
gt_weights = np.random.randint(1, 10, size=dim_in)
=======
dim_in = 100 
dim_outs = [1]
mlp = MLP(dim_in, dim_outs, non_linear_activation=True)
data = np.random.randint(1, 10, size=(20000, 100))
gt_weights = np.random.randint(1, 10, size=(100)).tolist()
>>>>>>> Stashed changes
gt_bias = 10

y = np.matmul(data, gt_weights) + 10
data = data.tolist()
y = y.tolist()
EPOCHS = 20
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
<<<<<<< Updated upstream
    ave_loss /= data_points
    print(f'{epoch}:{ave_loss}')
    plt_loss.append(ave_loss)
    if ave_loss < 0.001:
        break
plt.plot(np.arange(0, len(plt_loss)), plt_loss)
=======
    ave_loss /= 2000
    plt_loss.append(ave_loss) 
    print(epoch, ":", ave_loss)
plt.plot(np.arange(1, EPOCHS + 1), plt_loss) 
>>>>>>> Stashed changes
weights = [v.data for v in mlp.layers[0].neurons[0].w]
bias = mlp.layers[0].neurons[0].b.data
print('weights: ', weights, 'bias: ', bias)
print('gt_weights: ', gt_weights, 'bias: ', gt_bias)
<<<<<<< Updated upstream
errors = sum((weights-gt_weights))
=======
weights += [bias]
gt_weights += [gt_bias]
errors = sum((np.array(weights) - np.array(gt_weights))**2)
print(errors)
>>>>>>> Stashed changes
plt.show()
