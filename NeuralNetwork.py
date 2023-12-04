import random
import Tensor as t


class Module:
    def zero_grad(self):
        for p in self.params():
            p.grad = 0

    def params(self):
        raise NotImplementedError


class Neuron(Module):
    def __init__(self, dim_in, non_linear_activation=False, name=None):
        self.w = [t.Tensor(random.uniform(-1, 1),
                           name=f'w{idx}') for idx in range(dim_in)]
        self.b = t.Tensor(0, name='b')
        self.name = name
        self.non_linear_activation = non_linear_activation

    def __call__(self, x):
        for idx, xi in enumerate(x):
            if isinstance(xi, t.Tensor):
                xi.name = f'x{idx}'
            else:
                x[idx] = t.Tensor(xi, name=f'x{idx}')
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.non_linear_activation else act

    def params(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neutron(name={self.name},non_linear={self.non_linear_activation})"


class Layer(Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        self.neurons = [Neuron(dim_in, name=f"n{idx}", **kwargs)
                        for idx in range(dim_out)]

    def __call__(self, x):
        out = [nn(x) for nn in self.neurons]
        return out

    def params(self):
        return [p for n in self.neurons for p in n.params()]

    def __repr__(self):
        return f'Layer({", ".join(str(n) for n in self.neurons)})'


class MLP(Module):
    def __init__(self, dim_in: int, dim_outs: list[int], **kwargs):
        dimensions = [dim_in] + dim_outs
        self.layers = [Layer(dimensions[i], dimensions[i+1], **kwargs)
                       for i in range(len(dim_outs))]

    def __call__(self, x) -> list[t.Tensor]:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def params(self):
        return [p for layer in self.layers for p in layer.params()]

    def __repr__(self):
        return f'MLP of [{", ".join(str(layer) for layer in self.layers)}]'
