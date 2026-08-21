from typing import List, Optional
import torch

class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        return []
    
    def train(self):
        self.training = True

    def eval(self):
        self.training = False

class Linear(Module):
    def __init__(self, fan_in: int, fan_out: int, bias: bool = True, generator: Optional[torch.Generator] = None):
        # Xvaier initialize weights
        self.weight = torch.randn((fan_in, fan_out), generator=generator) / (fan_in ** 0.5)
        # Initialize bias as zeroes if bias is true
        self.bias = torch.zeros(fan_out) if bias else None
        self.out = None

    def forward(self, x: torch.Tensor):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        #running mean and var
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def forward(self, x: torch.Tensor):
        #compute batch norm in training mode
        if self.training:
            dim = 0 if x.ndim == 2 else (0, 1)
            mean = x.mean(dim=dim, keepdim=True)
            var = x.var(dim=dim, keepdim=True, unbiased=False)
        else:
            mean = self.running_mean
            var = self.running_var
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        self.out = self.gamma * x_norm + self.beta
        # Update running mean and bar
        if self.training:
            with torch.no_grad():
                # .squeeze() to remove all the 1 dims in mean，to keep running_mean as 1D Tensor (dim,)
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / (var + self.eps)
        self.out = self.gamma * x_norm + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh(Module):
    def __init__(self):
        self.out = None

    def forward(self, x: torch.Tensor):
        self.out = torch.tanh(x)
        return self.out

class Embedding(Module):
    def __init__(self, num_embeds: int, embed_dim: int, generator: Optional[torch.Generator] = None):
        self.weight = torch.randn((num_embeds, embed_dim), generator=generator)
        self.out = None

    def forward(self, x: torch.Tensor):
        self.out = self.weight[x]
        return self.out

    def parameters(self):
        return [self.weight]

class Flatten(Module):
    def __init__(self):
        self.out = None

    def forward(self, x: torch.Tensor):
        self.out = x.view(x.shape[0], -1)
        return self.out

class FlattenConsecutive(Module):
    def __init__(self, n: int):
        self.n = n
        self.out = None

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

class Sequential(Module):
    def __init__(self, layers: List[Module]):
        self.layers = layers
        self.out = None

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def train(self):
        for layer in self.layers:
            layer.train()

    def eval(self):
        for layer in self.layers:
            layer.eval()