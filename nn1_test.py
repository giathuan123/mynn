from engine.nn_v1.NeuralNetwork import MLP
from matplotlib import pyplot as plt
import numpy as np


# init neural networks
dim_in = 100 # in dimension
dim_outs = [1] # linear regression 
mlp = MLP(dim_in, dim_outs) 

# Ground truth
gt_weights = np.random.randint(1, 10, size=(dim_in)).tolist()
gt_bias = 10

# Generate ground truth data
data_points = 20000
data = np.random.randint(1, 10, size=(data_points, dim_in))

y = np.matmul(data, gt_weights) + gt_bias
data = data.tolist()
y = y.tolist()

# Hyper-parameters
EPOCHS = 20
learning_rate = 0.0001
plt_loss = []

# training loop
for epoch in range(EPOCHS):
    ave_loss = 0
    for xi, value in zip(data, y):
        mlp.zero_grad()
        out = mlp(xi)[0]
        loss = (value - out)**2 # MSE
        loss.backward() # get gradient
        ave_loss += loss.data
        for p in mlp.params(): # similar to step in pytorch
            p.data -= learning_rate * p.grad
    ave_loss /= data_points
    print(f'{epoch}:{ave_loss}')
    plt_loss.append(ave_loss)
    if epoch > 0 and epoch % 10 == 0:
        learning_rate *= 0.1
    if ave_loss < 0.0001:
        break
    
# Measuring performance
plt.plot(np.arange(0, len(plt_loss)), plt_loss)
weights = [v.data for v in mlp.layers[0].neurons[0].w]
bias = mlp.layers[0].neurons[0].b.data
weights = np.array(weights + [bias])
gt_weights = gt_weights + [gt_bias]
errors = np.sum((weights-gt_weights)**2)
print('weights:', weights)
print('gt_wieghts:', gt_weights)
print("Errors: ", errors)
