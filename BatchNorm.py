from OptTensor import OptTensor
import numpy as np


class BatchNorm:
    def __init__(self):
        self.gamma = None
        self.gamma_grad = None
        self.beta = None
        self.beta_grad = None
        self.mean_moving = 1
        self.mean_std = 0
        self.alpha = 0.95

    def eval(self, data):
        norm = (data.data - self.mean_moving) / self.mean_std
        return OptTensor(norm, children=data, backward_func='BatchNorm')

    def __call__(self, data: OptTensor):
        if not self.gamma:
            self.gamma = np.ones(data.shape[1])
        if not self.beta:
            self.beta = np.zeros(data.shape[1])
        norm = data.norm()
        data.out = norm(self.gamma) + self.beta
        self.meaning_moving = self.alpha + \
            (1-self.alpha)*data.data.mean(axis=0)
        self.mean_std = self.alpha + (1-self.alpha)*data.data.std(axis=0)

        def backward():
            # y = v*X_mean + b
            # dy/dv = X_mean
            # dy/db = 1
            # Application of the chain rule
            # gamma_grad = dL/dv = dL/dy * dy/dv
            # gamma_grad = dL/dv = dL/dy * X_mean
            # bias_grad = dL/db = dL/dy * dy/db
            # bias_grad = dL/db = dL/dy *
            # data.grad: dL/dx = dL/dy * dy/dx
            pass
        return None
