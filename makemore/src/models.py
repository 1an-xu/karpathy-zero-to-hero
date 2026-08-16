import torch
from typing import Optional
from src.nn import Sequential, Embedding, Tanh, BatchNorm1d, FlattenConsecutive, Linear, Flatten

def create_wavenet(
    embedding_size: int = 8,
    vacab_size: int = 27,
    hidden_dim: int = 124,
    generator: Optional[torch.Generator] = None
) -> Sequential:
    # 6 layers MLP
    model = Sequential([ 
        # construct embeddings
        Embedding(vacab_size, embedding_size, generator=generator),
        # input layer 1
        # input dim (B, block_size, emb_size), output dim (B, block_size / 2, hidden_dim)
        FlattenConsecutive(2),
        Linear(embedding_size * 2, hidden_dim, bias=False, generator=generator), 
        BatchNorm1d(hidden_dim),
        Tanh(),
        # input layer 2
        # input dim (B, block_size / 2, hidden_dim), output dim (B, block_size / 4, hidden_dim)
        FlattenConsecutive(2),
        Linear(hidden_dim * 2, hidden_dim, bias=False, generator=generator), 
        BatchNorm1d(hidden_dim),
        Tanh(),
        # input layer 3
        # input dim (B, block_size / 4, hidden_dim), output dim (B, hidden_dim)
        FlattenConsecutive(2),
        Linear(hidden_dim * 2, hidden_dim, bias=False, generator=generator), 
        BatchNorm1d(hidden_dim),
        Tanh(),
        # output layer, no tanh
        Linear(hidden_dim, vacab_size, generator=generator),
    ])

    # Adjust initialized parameters
    with torch.no_grad():
        # output layer: make less confident to avoid initial loss too large
        model.layers[-1].weight *= 0.1
        # apply tanh gain to weights of all other layers
        for layer in model.layers[: -1]:
            if isinstance(layer, Linear):
                layer.weight *= 5/3

    for p in model.parameters():
        p.requires_grad = True

    return model

def create_deep_mlp(
    embedding_size: int = 10,
    vacab_size: int = 27,
    block_size: int = 3,
    hidden_dim: int = 100,
    num_hidden_layers: int = 4,
    generator: Optional[torch.Generator] = None
) -> Sequential:
    in_dim = embedding_size * block_size

    # 6 layers MLP
    layers = [
        # construct embeddings
        Embedding(vacab_size, embedding_size, generator=generator),
        Flatten(),
        # input layer
        Linear(in_dim, hidden_dim, bias=False, generator=generator), 
        BatchNorm1d(hidden_dim),
        Tanh(),
    ]
    for _ in range(num_hidden_layers):
        layers.extend([
            # hidden layer 1
            Linear(hidden_dim, hidden_dim, bias=False, generator=generator),
            BatchNorm1d(hidden_dim),
            Tanh(),
        ])
    layers.extend([
        # output layer, no tanh
        # in production no batch norm as well, just Linear(hidden_dim, out_dim)
        Linear(hidden_dim, vacab_size, bias=False, generator=generator),
        BatchNorm1d(vacab_size),
    ])
    model = Sequential(layers)

    # Adjust initialized parameters
    with torch.no_grad():
        # output layer: make less confident to avoid initial loss too large
        layers[-1].gamma *= 0.1
        # apply tanh gain to weights of all other layers
        for layer in layers[: -1]:
            if isinstance(layer, Linear):
                layer.weight *= 5/3

    for p in model.parameters():
        p.requires_grad = True
        
    return model