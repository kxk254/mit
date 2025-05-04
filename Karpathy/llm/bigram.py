import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt


#hyper parameters
batch_size = 8 #64
block_size = 4 #256
max_iters = 5000
eval_interval = 250
learning_rate = 3e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 256 #384
n_head = 4  #6
n_layer = 4  #6
dropout = 0.2
#-------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

#create mapping
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  #(B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  #(B, T, T)
        wei = F.softmax(wei, dim=-1)  #(B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  #(B, T, C)
        out = wei @ v  #(B, T, T), @ (B,T,C) --> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #(B, T, C)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearly """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  #projection layer
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head the bumber of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)  #apply before 
        self.ln2 = nn.LayerNorm(n_embd)  #applu before not later which deviates from the paper
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.sa_head = MultiHeadAttention(4, n_embd//4) # 4 heads of 8-domentional self-attention
        # self.ffwd = FeedForward(n_embd)
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd),
        # )
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        #idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  #(B, T, C) batch, block, channel or vocab
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  #(T, C)
        x = tok_emb + pos_emb  #(B, T, C)
        # x = self.sa_head(x)  # apply one head of selt-attention (B,T,C)
        # x = self.ffwd(x)  #(B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  #(B, T, C)  but C is vocab_size channel

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # print("logits shape", logits.shape)
            # print("target shape", targets.shape)
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # print("after logits shape", logits.shape)
            # print("after target shape", targets.shape)

            loss = F.cross_entropy(logits, targets)
        return logits, loss
        
    def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
        for i in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predicion
            logits, loss = self(idx_cond)  #[1, 1, 65]
            # focus only on the last time step
            logits1 = logits[:, -1, :] #becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits1, dim=-1)  #(B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat ((idx, idx_next), dim = 1)  #(B, T+1)
            # print("logits", logits.shape)
            # print("logits1", logits1)
            # print("probs", probs)
            # print("idx_next", idx_next)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# create a pytorch optimiser
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:4f}, val loss {losses['val']:.4f}")
    
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


# PS D:\dev\mit\Karpathy\llm> py .\bigram.py
# step 0: train loss 4.284884, val loss 4.2823
# step 500: train loss 1.976354, val loss 2.0689
# step 1000: train loss 1.554858, val loss 1.7411
# step 1500: train loss 1.397579, val loss 1.6011
# step 2000: train loss 1.304354, val loss 1.5458
# step 2500: train loss 1.241086, val loss 1.5099
# step 3000: train loss 1.187475, val loss 1.4963
# step 3500: train loss 1.138522, val loss 1.4905
# step 4000: train loss 1.095621, val loss 1.4935
# step 4500: train loss 1.046936, val loss 1.5009

# I will lie on our larks, this sword passess
# And them our knees of rocks a great treasons with
# A she-grief orde and a senator, this quare is enemies.
# That nature seen we banks; now mine shall hear me,
# For I ever used that trod is so into express,
# That I have write out a thousand house
# And can light of a viney's kiss again:
# My mother's pare--if I confess nhugs for this:
# It that set my breast, I lorder say,
# Thereug, my sweets, whose raises grant
# But they came to themselves again. By ears?

# CLADY:
# T