import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
    block_size: int = 3
    vocab_size: int = 26
    n_embed: int = 64
    n_embed2: int = 64


class MLP(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        #token embeddings table
        self.wte = nn.Embedding(
            config.vocab_size + 1, 
            config.n_embed)
        self.mlp = nn.Sequential(
            #layer 1: input dimension = block_size * n_embed, hidden dimension = n_embed2
            nn.Linear(self.block_size * config.n_embed, config.n_embed2),
            # tanh activation on layer 1
            nn.Tanh(),
            # layer 2: input dimension = n_embed2, output dimension = vocab_size
            nn.Linear(config.n_embed2, self.vocab_size)
        )

    #idx: training data x: (b, t)
    #target: training data y: (b, t)
    def forward(self, idx, target=None):
        # used to construct x only. (block_size, b, t, n_embed)
        embs = [] 
        for i in range(self.block_size):
            #take out embeddings from table for idx. (b, t, n_embed)
            token_emb = self.wte(idx)
            #shift idx to right by one index
            idx = torch.roll(idx, 1, 1)
            #fill the first index with special <BLANK> token
            idx[:, 0] = self.vocab_size
            embs.append(token_emb)
        #concat the elements in embs at last dimension 
        #x: (b, t, n_embed * block_size)
        x = torch.cat(embs, -1)
        #logits: (b, t, vocab_size)
        logits = self.mlp(x)

        loss = None
        if target is not None:
            loss = F.cross_entropy(
                #(b * t, vocab_size)
                logits.view(-1, self.vocab_size), 
                #(b * t,)
                target.view(-1), 
                #ignore -1 paddings in target
                ignore_index=-1)
        return logits, loss