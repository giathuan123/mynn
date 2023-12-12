from OptTensor import OptTensor
from numpy._typing import NDArray
import numpy as np

class BinaryCrossEntropyLoss:
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, data: OptTensor, label):
        assert data.shape[1] == 2
        assert label.shape[0] == data.shape[0]
        one_hot_labels = np.zeros_like(data.data) # (d, C)
        for idx, l in enumerate(label):
            one_hot_labels[idx, l] = 1
        losses =  np.random.randn(data.shape[0])
        loss = np.array(float('inf'))
        if self.reduction == "mean":
            loss: NDArray = losses.mean()
        elif self.reduction == "sum":
            loss: NDArray = losses.sum()
        out_tensor = OptTensor(loss, children=data, backward_func='BinaryCrossEntropyLoss') # compute loss
        data.out = out_tensor
        def backward():
            assert out_tensor
            data.grad = out_tensor.grad
        out_tensor._backward = backward
        return out_tensor

