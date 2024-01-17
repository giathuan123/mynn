from OptTensor import OptTensor
from numpy._typing import NDArray
import numpy as np


def header(heading: str) -> str:
    return '-'*5 + heading + '-'*5


class BinaryCrossEntropyLoss:
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, data: OptTensor, label: NDArray):

        assert data.shape[1] == 2
        assert label.shape[0] == data.shape[0]
        assert np.allclose(data.data.sum(axis=1), 1)
        assert (np.logical_or(label == 0, label == 1)).all()

        # creating the one hot vector
        one_hot_labels = np.zeros_like(data.data)  # (d, C)
        for idx, l in enumerate(label):
            one_hot_labels[idx, l] = 1

        # calculating loss
        losses = -(np.multiply(one_hot_labels,
                   np.log(data.data + 10**-100)).sum(axis=1))
        # using mean reduce
        loss = losses.mean()
        # creating out tensor
        out_tensor = OptTensor(
            loss, children=data, backward_func='BinaryCrossEntropyLoss')  # compute loss
        data.out = out_tensor

        def backward():
            assert out_tensor
            jacobian = np.where(
                one_hot_labels == 1, -np.reciprocal(data.data + 10**-100), np.zeros_like(data.data))
            data.grad = (jacobian * out_tensor.grad /
                         data.shape[0])  # assume out tensor is 1
        out_tensor._backward = backward
        return out_tensor
