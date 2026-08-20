import torch

class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, tokens: list[int]):
        return "".join([self.itos[i] for i in tokens])


def split_data(
        tokenizer: CharTokenizer, 
        text: str, 
        split: float = 0.9, 
        device: str | None = None):
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)
    
    n = int(split * len(tokens))
    return tokens[: n], tokens[n :]

def get_batches(
        batch_size: int, 
        block_size: int, 
        dataset: torch.Tensor, 
        device: str | None = None):
    #use the same device as dataset if not specified
    dev = dataset.device if device is None else torch.device(device)
    #select random batch_size indexes from [0, dataset_size - block_size]
    ix = torch.randint(0, len(dataset) - block_size, (batch_size, ), device=dev)
    X = torch.stack([dataset[i: i + block_size] for i in ix])
    Y = torch.stack([dataset[i + 1: i + block_size + 1] for i in ix])
    return X, Y