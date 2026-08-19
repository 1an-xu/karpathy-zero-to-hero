from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# V1 Bigram model - No Transformer
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets = None):
        # x: (B, T), target: (B, T)
        logits = self.token_embedding_table(x)

        if targets == None:
            loss = None
        else:
            # logits: (B, T, vocab_size)
            B, T, C = logits.shape
            logits = logits.view(B * T, C) # or logits.view(-1, C)
            targets = targets.view(B * T) # or target.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx: (B, T)
        # generate one token each time
        for _ in range(max_new_tokens):
            #logits: (B, T, vocab_size)
            logits, _ = self(idx)
            # only need to look at the prediction at last token in the sequence
            # (B, T, vocab_size) -> (B, vocab_size)
            logits = logits[:, -1, :]
            # softmax/normalize the last dimension: (B, softwax(C))
            props = F.softmax(logits, -1)
            # predict next token: (B, 1)
            idx_next = torch.multinomial(props, num_samples=1)
            # append next token to idx
            # idx: (B, T), idx_next: (B, 1) -> idx: (B, T + 1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# Single head attention
class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
        self.dropout = nn.Dropout(dropout)
        self.out = None

    def forward(self, x):
        B, T, C = x.shape 
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5) 
        # tril: (block_size, block_size) -> (T, T), when T <= block_size
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei) # Dropout 20% after softmax
        v = self.value(x) # v: (B, T, head_size)
        self.out = wei @ v # out: (B, T, head_size)
        return self.out


# V2 Biagram model - with Single Head Attention
class BigramLanguageModelV2(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # keep head_size = n_embd
        self.sa_head = Head(n_embd, n_embd, block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, x, targets = None):
        # x: (B, T), target: (B, T)
        B, T = x.shape
        # embeddings for each token
        token_emb = self.token_embedding_table(x) # (B, T, n_embd)
        # embedding for each position, same for all the batches
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, n_embd)
        x = token_emb + pos_emb # x: (B, T, n_embd)
        x = self.sa_head(x) # x: (B, T, n_embd), head_size=n_embd
        logits = self.lm_head(x) # logits: (B, T, vocab_size)

        if targets == None:
            loss = None
        else:
            # logits: (B, T, vocab_size)
            B, T, C = logits.shape
            logits = logits.view(B * T, C) # or logits.view(-1, C)
            targets = targets.view(B * T) # or target.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx: (B, T)
        # generate one token each time
        for _ in range(max_new_tokens):
            # take the last block_size of idx since pos embeddin table is block_size at max
            idx_updated = idx[:,-self.block_size:]
            #logits: (B, T, vocab_size)
            logits, _ = self(idx_updated)
            # only need to look at the prediction at last token in the sequence
            # (B, T, vocab_size) -> (B, vocab_size)
            logits = logits[:, -1, :]
            # softmax/normalize the last dimension: (B, softwax(C))
            props = F.softmax(logits, -1)
            # predict next token: (B, 1)
            idx_next = torch.multinomial(props, num_samples=1)
            # append next token to idx
            # idx: (B, T), idx_next: (B, 1) -> idx: (B, T + 1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx