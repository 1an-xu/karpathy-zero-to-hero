import torch
from typing import List, Tuple, Dict

class CharTokenizer:
    def __init__(self, words: List[str]):
        chars = sorted(list(set("".join(words))))
        self.stoi = {char: i+1 for i, char in enumerate(chars)}
        self.stoi['.'] = 0
        self.itos = {i: s for s, i in self.stoi.items()}

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, indices: List[int]) -> str:
        return "".join([self.itos[i] for i in indices])


def build_dataset(words: List[str], tokenizer: CharTokenizer, block_size: int = 3):
    X = []
    Y = []
    for word in words: 
        context = [0] * block_size
        for ch in word + '.':
            index = tokenizer.stoi[ch]
            X.append(context)
            Y.append(index)
            context = context[1:] + [index]
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

def split_data(
        words: List[str], 
        tokenizer: CharTokenizer, 
        block_size: int = 8, 
        splits: Tuple[float, float] = (0.8, 0.9)):
    # Set up training/val/test data sets
    n1 = int(splits[0] * len(words))
    n2 = int(splits[1] * len(words))

    Xtr, Ytr = build_dataset(words[: n1], tokenizer, block_size)
    Xdev, Ydev = build_dataset(words[n1: n2], tokenizer, block_size)
    Xtest, Ytest = build_dataset(words[n2:], tokenizer, block_size)
    return (Xtr, Ytr), (Xdev, Ydev), (Xtest, Ytest)