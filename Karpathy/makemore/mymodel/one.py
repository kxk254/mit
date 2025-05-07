import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

words = open("../names.txt", 'r').read().splitlines() 
    
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

x_train, y_train = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for c1, c2 in zip(chs, chs[1:]):
        ix1 = stoi[c1]
        ix2 = stoi[c2]
        x_train.append(ix1)
        y_train.append(ix2)

x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)
num = x_train.nelement()

#initialize the 'network'
g = torch.Generator().manual_seed(667737)
# W = torch.randn((27, 27), generator=g, requires_grad=True)
linear = torch.nn.Linear(27, 27) 
batch_norm = torch.nn.BatchNorm1d(27)
optimizer = torch.optim.Adam(
    list(linear.parameters()) + list(batch_norm.parameters()), 
    lr=1.0, weight_decay=0.01  # L2 regularization included here
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

x_train_encoded = F.one_hot(x_train, num_classes=27).float()  # x_train_encoded torch.Size([228146, 27])
loss_fn = torch.nn.CrossEntropyLoss()

# Dummy forward pass to populate W.grad for first time
logits = linear(x_train_encoded)
loss = loss_fn(logits, y_train)
loss.backward()
w_grad = linear.weight.grad.detach().numpy()
w_data = linear.weight.data.detach().numpy()

# Enable interactive plotting
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
im_grad = ax1.imshow(w_grad, cmap="seismic", vmin=-1, vmax=1)
plt.colorbar(im_grad, ax=ax1)  # <-- pass `im` as the mappable object
ax1.set_title = ax1.set_title("gradient")

im_data = ax2.imshow(w_data, cmap="seismic", vmin=-1, vmax=1)
plt.colorbar(im_data, ax=ax2)  # <-- pass `im` as the mappable object
ax2.set_title = ax2.set_title("weight")

for step in range(100):

    # forward pass
    # logits = x_train_encoded @ W
    logits = linear(x_train_encoded)
    logits = batch_norm(logits)
    # probs = torch.softmax(logits, dim=1)

    # loss = -probs[torch.arange(num), y_train].log().mean()
    loss = loss_fn(logits, y_train)
    # weight decay
    # loss = loss + 0.01*(W**2).mean()
    # print(loss.item())

    # backward pass
    # W.grad = None
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # # update
    # W.data += -10 * W.grad

    # Visualization
    w_grad = linear.weight.grad.detach().numpy()
    w_data = linear.weight.data.detach().numpy()

    vmax_grad = abs(w_grad).max().item()
    im_grad.set_data(w_grad)
    im_grad.set_clim(-vmax_grad, vmax_grad)

    vmax_data = abs(w_data).max().item()
    im_data.set_data(w_data)
    im_data.set_clim(-vmax_data, vmax_data)  # Dynamically scale color range
    # plt.colorbar()
    plt.suptitle(f"Step {step}, Loss: {loss.item():.4f}")
    # plt.tight_layout()
    # plt.draw()
    plt.pause(0.1)  # Pause to make updates visible

    """
    WEIGHTS DECAY
    L2 regularization:
    Helps prevent overfitting by discouraging large weights
    Encourages the model to find simpler solutions
    Smooths the optimization landscape
    """

plt.ioff()  # Turn off interactive mode
plt.show()  # Show the final plot


for i in range(5):
    out = []
    ix= 0
    while True:
        x_out = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        # logits = x_out @ W
        logits = linear(x_out)
        p = torch.softmax(logits, dim=1)

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))


"""
✅ Interpreting Weights (weight.data):
Color	            Meaning	            Interpretation
Red/Blue	        Large weights	    Model is confident or memorizing
White	            Small/near-zero     weights	Model hasn't learned much yet / regularized
Too much Red/Blue	Exploding weights	Learning rate too high or poor regularization
All White (stuck)	No learning	        Weights not updating / dead layer

✅ Interpreting Gradients (weight.grad):
Color	            Meaning	                            Interpretation
Red/Blue	        Strong positive/negative gradient	Weights are being updated heavily
White	            Gradient near zero	                Either converged or no signal is flowing
All White early on	Vanishing gradients	                Bad architecture, poor initialization, or issue with activations (like saturation in ReLU/tanh)
"""