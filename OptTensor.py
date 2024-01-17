from numpy._typing import NDArray
from typing import Optional
import numpy as np


class OptTensor:
    def __init__(self, data: NDArray, children=None, backward_func=None, name=None):
        self.data = data
        self.grad = np.zeros_like(data)
        self.out: Optional[OptTensor]
        self.children = children
        self.backward_func = backward_func
        self._backward = lambda: None
        self.name = name

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f'OptTensor({self.shape}, backward_func={self.backward_func}, {"name=" + self.name if self.name else ""})'

    def print(self):
        current_node = self
        space = ' '
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

    def softmax(self, temperature=1):
        exp = np.exp(self.data/temperature)
        softmax = exp / np.expand_dims(exp.sum(axis=1), axis=1)
        self.out = OptTensor(softmax, children=self, backward_func="Softmax")

        def backward():
            assert self.out
            self.grad = np.zeros_like(self.data)
            for i in range(self.shape[0]):
                current_softmax = softmax[i].reshape((1, softmax.shape[1]))
                jacobian = np.diag(current_softmax.squeeze(0)) \
                    - current_softmax.T @ current_softmax
                self.grad[i] = jacobian.T @ self.out.grad[i]
        self.out._backward = backward
        return self.out

    def __add__(self, other):
        assert isinstance(other, OptTensor)
        data = self.data + other
        self.out = OptTensor(data, children=self, backward_func='Addition')

        def backward():
            assert self.out
            self.grad = self.out.grad
            other.grad = self.out.grad
        self.out._backward = backward
        return self.out

    def norm(self):
        raise NotImplementedError()

    def __call__(self, operand):
        if not isinstance(operand, OptTensor):
            operand = OptTensor(operand)
        self.out = OptTensor(operand.data @ self.data,
                             children=self,
                             backward_func='MatMul')

        def backward():
            assert self.out
            jacobian = operand.data
            operand.grad = self.out.grad @ self.data.T
            self.grad = jacobian.T @ self.out.grad
        self.out._backward = backward
        self.children = operand
        return self.out

    def relu(self):
        relu_out = np.maximum(0, self.data)
        self.out = OptTensor(relu_out, children=self, backward_func="ReLU")

        def backward():
            assert self.out
            self.grad = (relu_out > 0) * self.out.grad
        self.out._backward = backward
        return self.out
