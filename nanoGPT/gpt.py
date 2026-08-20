import torch
from data.prepare import CharTokenizer, split_data, get_batches
from model import GPTLanguageModel

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

# prepare dataset for training and eval
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
vocab_size = len(set(text))
tokenizer = CharTokenizer(text)
train_data, val_data = split_data(tokenizer, text, split=0.9, device=device)

# create model
model = GPTLanguageModel(
    vocab_size, 
    n_embd, 
    n_head, 
    block_size, 
    n_layer, 
    dropout).to(device)
print(f"Total params: {sum(p.nelement() for p in model.parameters()) / 1e6:.2f}M")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        dataset = train_data if split == 'train' else val_data
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            X, Y = get_batches(batch_size, block_size, dataset, device)
            _, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Train model
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    Xtr, Ytr = get_batches(batch_size, block_size, train_data, device)
    logits, loss = model(Xtr, Ytr)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate text
predicts = model.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0]
print(tokenizer.decode(predicts.tolist()))