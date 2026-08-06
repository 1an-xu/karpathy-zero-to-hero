import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._op = _op
        self._prev = set(_children)
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data} grad={self.grad} label={self.label})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            other.grad += 1 * out.grad
            self.grad += 1 * out.grad
        
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            other.grad += self.data * out.grad
            self.grad += other.data * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self * -1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only support power by float / int"
        out = Value(self.data ** other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'Relu')

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1.0 - t**2) * out.grad
        out._backward = _backward
        return out

                
    def __radd__(self, other): # 1 (other) + self
        return self + other

    def __rmul__(self, other): # 1 (other) * self
        return self * other

    def __rsub__(self, other): # 1(other) - self 
        return other + (-self)

    def backward(self):
        nodes = []
        visited = set()

        # DFS
        def topological_sort(node):
         if node not in visited:
             visited.add(node)
             for child in node._prev:
                 topological_sort(child)
             nodes.append(node)

        self.grad = 1.0
        topological_sort(self)
        for node in reversed(nodes):
            node._backward()