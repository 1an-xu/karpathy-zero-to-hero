import torch
from data.prepare import CharTokenizer, split_data, get_batches
from model import BigramLanguageModel, BigramLanguageModelV2, GPTLanguageModel

block_size = 8
n_embd = 32
batch_size = 32
num_steps = 10000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#prepare data
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
vocab_size = len(set(text))
tokenizer = CharTokenizer(text)
train_data, val_data = split_data(tokenizer, text, split=0.9, device=device)

#create model
#model = BigramLanguageModel(vocab_size)
model = BigramLanguageModelV2(vocab_size, n_embd, block_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#train model
for step in range(num_steps):
    Xtr, Ytr = get_batches(batch_size, block_size, train_data, device)
    logits, loss = model(Xtr, Ytr)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 1000 == 0: # print every once in a while
        print(f'{step:7d}/{num_steps:7d}: {loss.item():.4f}')

#generate text
predicts = model.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0]
print(tokenizer.decode(predicts.tolist()))