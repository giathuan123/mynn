from OptTensor import OptTensor
from numpy._typing import NDArray
import numpy as np

def header(heading: str) -> str:
    return '-'*5 + heading + '-'*5
    
class BinaryCrossEntropyLoss:
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, data: OptTensor, label: NDArray):
        
        assert data.shape[1] == 2 # binary class
        assert label.shape[0] == data.shape[0] # 1 label for 1 example
        assert np.allclose(data.data.sum(axis=1), 1) # assume data have gone through the softmax function
        assert (np.logical_or(label == 0, label==1)).all() # label only have 2 classes
        
        # creating the one hot vector
        one_hot_labels = np.zeros_like(data.data) # (d, C)
        for idx, l in enumerate(label):
            one_hot_labels[idx, l] = 1
            
        # calculating loss
        losses = -(np.multiply(one_hot_labels, np.log(data.data)).sum(axis=1))
        # using mean reduce
        loss = losses.mean()
        # creating out tensor 
        out_tensor = OptTensor(loss, children=data, backward_func='BinaryCrossEntropyLoss') # compute loss
        data.out = out_tensor
        def backward():
            assert out_tensor
            jacobian = -np.multiply(one_hot_labels, np.reciprocal(data.data + 0.0001))
            data.grad = jacobian * out_tensor.grad
        out_tensor._backward = backward
        return out_tensor

loss = BinaryCrossEntropyLoss()
label: NDArray = np.random.randint(0, 2, size=(300,))
x = OptTensor(np.zeros((label.shape[0], 2)))
prob = x.softmax()

itr = 0
current_loss = None
prob = None
while(itr < 2):
    prob = x.softmax()
    current_loss = loss(prob, label)
    current_loss.backward()
    x.data -= 0.01*x.grad
    # print(header('probs'))
    # print(prob.data)
    # print(header('label'))
    # print(label)
    # print('Current Loss', current_loss.data)
    itr += 1

assert prob
print(current_loss)
print('correct' if (prob.data.argmax(axis=1) == label).all() else 'not correct')
