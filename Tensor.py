from graphviz import Digraph
from matplotlib import pyplot as plt


class Tensor:
    ADD = '+'
    MULT = '*'
    RELU = 'ReLU'

    def __init__(self, data, name=None, _graph=(), _op=''):
        self.data = data
        self.grad = 0
        self.name = name
        self._op = _op
        self._prev = set(_graph)
        self._backward = lambda: None

    def zero_grad(self):
        self.grad = 0
        for node in self._prev:
            node.zero_grad()

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def relu(self):
        out = Tensor(0 if self.data < 0 else self.data,
                     name="Relu", _graph=(self,), _op=Tensor.RELU)

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, name=f'{self.name}+{other.name}',
                     _graph=(self, other), _op=Tensor.ADD)

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Tensor(self.data**other, _graph=(self,), _op=f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        if not hasattr(self, 'topo'):
            build_topo(self)
            self.topo = topo

        self.grad = 1
        for v in reversed(self.topo):
            v._backward()

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     name=f'{self.name}*{other.name}', _graph=(self, other), _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def plot(self, dot=None, parent=None):
        if not dot:
            dot = Digraph(format='svg', graph_attr={'rankdir': 'TB'})

        node_id = str(id(self))
        dot.node(name=node_id, label='{%s | data %.4f | grad %.4f }' %
                 (self.name, self.data, self.grad), shape='record')
        if self._op:
            dot.node(name=node_id + self._op, label=self._op)
            dot.edge(node_id + self._op, node_id)
        if parent:
            dot.edge(node_id, str(id(parent)) + parent._op)
        for node in self._prev:
            node.plot(dot, parent=self)
        return dot

    def __neg__(self):
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1
