from engine import Value
import random

class Neuron:
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1), label=f'w{i}') for i in range(nin)]
        self.b = Value(random.uniform(-1, 1), label='b') 
        self.nonlin = nonlin

    def __call__(self, x):
        result = sum((w1 * x1 for w1, x1 in zip(self.w, x)), self.b)
        # use tanh for now to avoid dead ReLU problem
        return result.tanh()
        # apply relu for non linear mapping
        #return result.relu() if self.nonlin else result

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        result = [neuron(x) for neuron in self.neurons]
        return result[0] if len(result) == 1 else result

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        dims = [nin] + nouts
        self.layers = [Layer(dims[i], dims[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]