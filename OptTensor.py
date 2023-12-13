from numpy._typing import NDArray
from typing import Optional
import numpy as np


class OptTensor:
    def __init__(self, data: NDArray, children=None, backward_func=None):
        self.data = data
        self.grad = np.zeros_like(data)
        self.out: Optional[OptTensor]
        self.children = children
        self.backward_func = backward_func
        self._backward = lambda: None

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f'OptTensor({self.shape}, backward_func={self.backward_func})'

    def print(self):
        current_node = self
        space = '\t'
        level = 0
        while current_node:
            print(space*level + repr(current_node))
            level += 1
            current_node = current_node.children

    def backward(self):
        self.grad = np.ones_like(self.data)
        current_node = self
        while current_node:
            current_node._backward()
            current_node = current_node.children

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def softmax(self):
        exp = np.exp(self.data)
        softmax = exp / np.expand_dims(exp.sum(axis=1), axis=1)
        self.out = OptTensor(softmax, children=self, backward_func="Softmax")

        def backward():
            assert self.out
            self.grad = np.zeros_like(self.data)
            for i in range(self.shape[0]):
                current_softmax = softmax[i].reshape((1, softmax.shape[1]))
                jacobian = np.diag(current_softmax.squeeze(0)) \
                    - current_softmax @ current_softmax.T
                self.grad[i] = jacobian.T @ self.out.grad[i]
        self.out._backward = backward
        return self.out

    def __add__(self, other):
        assert isinstance(other, OptTensor)
        data = self.data + other
        self.out = OptTensor(data, children=self, backward_func='Addition')

        def backward():
            assert self.out
            self.grad += self.out.grad
            other.grad += self.out.grad
        self.out._backward = backward
        return self.out

    def norm(self):
        raise NotImplementedError()

    def __call__(self, operand):
        self.out = OptTensor(operand @ self.data,
                             children=self, backward_func='MatMul')

        def backward():
            assert self.out
            jacobian = operand
            self.grad += jacobian.T @ self.out.grad
        self.out._backward = backward
        return self.out

    def relu(self):
        relu_out = np.maximum(0, self.data)
        self.out = OptTensor(relu_out, children=self, backward_func="ReLU")

        def backward():
            assert self.out
            self.grad = (relu_out > 0) * self.out.grad
        self.out._backward = backward
        return self.out
