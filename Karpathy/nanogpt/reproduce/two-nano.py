from dataclasses import dataclass 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import inspect
import matplotlib.pyplot as plt

# env\Scripts\activate
# cd .\Karpathy\nanogpt\reproduce\ 

class CasualSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        """ 
        different heads can capture different kinds of information about the input sequence. 
        By combining them, the model can learn a more rich, nuanced representation of the input.
        Head 1 might learn to focus on local context (i.e., nearby words or tokens).
        Head 2 might focus on long-range dependencies (i.e., distant tokens).
        Head 3 might focus on syntax (i.e., word order and structure).
        Head 4 might focus on semantic meaning (i.e., word meanings and relationships).

        head dimension (head_dim) refers to the size of the vectors used by each head.
        This means each head works on smaller subspaces of the original data.
        Each T x T matrix (for each head) represents the attention scores between each query token and key token in the sequence.        
        """
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k ,v = qkv.split(self.n_embd, dim=2)
        # print("q", q.size(), q[0][0], "k", k[0][0], "v", v[0][0],)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, head, T, headsize)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # print("y", y.size(), y, )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # print("y aft", y.size(), y, )
        y = self.c_proj(y)
        return y
    
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass 
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 6 #12
    n_head: int = 8 #12
    n_embd: int = 8 #768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config 

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd), 
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):

        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of the length {T}, block size of {B}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  #(T)
        pos_emb = self.transformer.wpe(pos) #(T, n_emb)
        tok_emb = self.transformer.wte(idx)  #(B, T, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretraied GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weight from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),   #124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),  #350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),  #774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),  #124M params
        }[model_type]
        config_args['vocab_size'] = 50257  
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] 

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # ignore these, just a buffer
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                # if sd_hf[k].shape[::-1] != sd[k].shape:
                #     print(f"Shape mismatch for key {k}: HF shape {sd_hf[k].shape}, expected {sd[k].shape[::-1]}")
                #     continue  # or handle appropriately
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    
def get_most_likely_row(tokens, mask, logits):
    """ 
    Which sequence was most likely, according to the model?
    - Shifts logits and tokens to align predictions with targets.
    - Computes cross-entropy loss per token (how wrong was the model).
    - Applies the mask to ignore padding or irrelevant tokens.
    - Averages the loss over valid tokens → gives average loss per sequence.

    """
    print("get_most_likely_row")
    shift_logits = (logits[..., :-1, :]).contiguous()  #(B, T-1, Vocabsize)
    shift_tokens = (tokens[..., 1:]).contiguous()  #(B, T-1)
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))  #(B*T-1, V)
    flat_shift_tokens = shift_tokens.view(-1)  #(B*T-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none') #(B*T-1)
    shift_losses = shift_losses.view(tokens.size(0), -1)  #(B, T-1)
    shift_mask = (mask[..., 1:]).contiguous()  #(B, T-1)
    masked_shift_losses = shift_losses * shift_mask    #(B, T-1)
    sum_loss = masked_shift_losses.sum(dim=1)  #(B, )
    avg_loss = sum_loss / shift_mask.sum(dim=1)  #(B, )
    pred_norm = avg_loss.argmin().item()  #integer    Find the index of the row with the lowest average loss 
    return pred_norm

#--------DETECT THE DEVICE ---------------
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
# device = 'cpu'
print(f"using device: {device}")
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
#--------DETECT THE DEVICE ---------------

### shakespear
import tiktoken 
import math, time

class DataLoaderLite:
        
    def __init__(self, B, T, split):
        self.B = B 
        self.T = T
        self.split = split
        text = open('../input.txt', 'r').read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)        

        self.tokens = torch.tensor(tokens)
        # print(f"loaded {len(self.tokens)} tokens")
        # print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        self.total_tokens =len(self.tokens)

        self.current_position = 0
        self.max_position = self.total_tokens - (B * T + 1)
        self.batches_per_epoch = self.max_position // (B*T)
        print(f"[{self.split}] Loaded {self.total_tokens} tokens from length {len(text)}")
        print(f"[{self.split}] 1 epoch = {self.batches_per_epoch} batches")

        # self.count = 0  #debug
    
    def next_batch(self):
        B, T = self.B, self.T
        if self.current_position > self.max_position:
            # raise StopIteration("Epoch complete") 
            return None

        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        # print(f"position: [{self.current_position}], count: {self.count}")
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B*T
        # self.count+=1  #debug
        return x, y
    

## GRAPH================================

class ActivDist:

    def __init__(self, W, H):
        self.figsize = (W, H)
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.legends = []
    
    def figure(self, layers, step):
        self.ax.clear()
        self.legends.clear()

        for i, layer in enumerate(layers):
            if isinstance(layer, nn.GELU):
                t = layer.out
                print('layer %d(10%s): mean %+.2f, std %.2f, satured: %2.f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
                hy, hx = torch.histogram(t, density=True)
                self.ax.plot(hx[:-1].detach(), hy.detach())
                self.legends.append(f'layer {i} ({layer.__class__.__name__})')
        self.ax.legend(self.legends)
        self.ax.set_title(f'{step} - activation distribution')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

activations = {}
gradients = {}

# def save_activation(name):
#     def hook(module, input, output):
#         activations[name] = output.detach()
#         module.out = output.detach()
#     return hook

def save_activation_and_gradients(name):
    def hook(module, input, output):
        # Save the activations (during forward pass)
        activations[name] = output.detach()

        # Save the gradients (during backward pass)
        def save_grad(grad):
            gradients[name] = grad.detach()

        # Register backward hook to capture gradients of activations
        output.register_hook(save_grad)


    return hook

##------------------------------------
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

max_lr = 0.01 #6e-4
min_lr = max_lr * 0.1
warmup_steps = 100
max_steps = 1001
enc = tiktoken.get_encoding('gpt2')

total_batch_size = 2 * 32
B = 2
T = 16
grad_accum_steps = total_batch_size // (B * T)
train_loader = DataLoaderLite(B=B, T=T, split="train")  #4x32
val_loader = DataLoaderLite(B=B, T=T, split="val")  #4x32


def get_lr(it):
    if it < warmup_steps:
        # x = max_lr * (it+1) / warmup_steps
        # print("ir warmup", x)
        return max_lr * (it+1) / warmup_steps
    
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for i, layer in enumerate(model.modules()):
    if isinstance(layer, nn.GELU):
        layer.register_forward_hook(save_activation_and_gradients(f'gelu_{i}'))

plt.ion()
pltactivation = ActivDist(16, 5)

for i in range(max_steps):
    t0 = time.time()
    last_step = (i == max_steps -1)

    model.train()
    optimizer.zero_grad()
    loss_accum = 0
   
    for micro_step in range(grad_accum_steps):
        batch = train_loader.next_batch()
        if batch is None:
            print("Epoch complete — stopping training or restarting loader")
            break
        x, y = batch
        x, y = x.to(device), y.to(device)
        

        with torch.autocast(device_type=device, dtype=torch.float16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()


    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()    
    dt = (t1-t0)*1000
    if dt == 0:
        dt = 3e-8
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt 
    if i % 100 == 0:
        print(f"step: {i} |loss {loss_accum} | lr={lr:.4f} | dt: {dt:.2f} | token processed: {tokens_processed} | tokens per sec: {tokens_per_sec:.4f}")
               
        model.eval()
        with torch.no_grad():
            model(x, y)
        
        pltactivation.figure(model.modules(), i)
        plt.pause(0.01)

plt.ioff()
plt.show()