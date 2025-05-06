import torch.nn as nn
import torch
from subfunc import data_loader
import random
import matplotlib.pyplot as plt

random.seed(1337)
torch.manual_seed(1337)

x, y, x_val, y_val, x_test, y_test = data_loader()
x = torch.tensor(x)
y = torch.tensor(y)
x_val = torch.tensor(x_val)
y_val = torch.tensor(y_val)
x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test)
print("y shape", y.shape)
steps = 700
sample, yoko, tate = x.shape
h1 = 80
h2 = 30
in_shape = yoko*tate
l1 = nn.Linear(in_shape,h1)
# nl1 = nn.LeakyReLU(0.2)
bn1 = nn.BatchNorm1d(h1)
nl1 = nn.ReLU()
# nl1 = nn.Softmax(dim=1)
l2 = nn.Linear(h1, h2)
bn2 = nn.BatchNorm1d(h2)
nl2 = nn.ReLU()
# nl2 = nn.Softmax(dim=1)
outl = nn.Linear(h2, 10)
input_data = torch.reshape(x, (sample, yoko*tate))
sample_val, yoko, tate = x_val.shape
x_val = torch.reshape(x_val, (sample_val, yoko*tate))

# optimizer and loss
params = list(l1.parameters()) + list(l2.parameters()) + list(outl.parameters())
lr=0.01
optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
loss_fn = nn.CrossEntropyLoss()

for step in range(steps):

    output1 = l1(input_data)
    output1 = bn1(output1)  #batch normal
    output2 = nl1(output1)  #non lenear 1
    output3 = l2(output2)
    output3 = bn2(output3)  #batch normal
    output4 = nl2(output3) #non lenear 2
    output = outl(output4)

    if step == 60:
         lr = lr * 0.5
         for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    if step == 110:
         lr = lr * 0.4
         for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    if step == 280:
         lr = lr * 0.5
         for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    if step == 450:
         lr = lr * 0.4
         for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    if step == 550:
         lr = lr * 0.2
         for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    loss_output = loss_fn(output, y)
    if step % 10 == 0:
        with torch.no_grad():

            # accuracy 
            preds = torch.argmax(output, dim=1)
            correct = (preds == y).sum().item()
            accuracy = correct / y.size(0)

            # validation
            output_val = l1(x_val)
            output_val = bn1(output_val)
            output_val = nl1(output_val)
            output_val = l2(output_val)
            output_val = bn2(output_val)
            output_val = nl2(output_val)
            output_val = outl(output_val)

            # validation loss
            val_loss = loss_fn(output_val, y_val)

            # calculate validation accuracy
            preds_val = torch.argmax(output_val, dim=1)
            correct_val = (preds_val == y_val).sum().item()
            val_accuracy = correct_val / y_val.size(0)

        print(f"{step:2d} iter -> loss output :{loss_output.item():.4f} | lr: {lr:7f} | Accuracy: {accuracy*100:.2f}%| Val Loss: {val_loss.item():.4f} | Val Acccuracy: {val_accuracy*100:.2f}%")
    
    loss_output.backward()
    optimizer.step()
    

