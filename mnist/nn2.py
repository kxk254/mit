from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt

"""
w=weights, b=bias, i = input, h=hidden, o=output, l=label
e.g. w_i_h = weights from input layer to hidden layer
images = (60000, 784)
label = (60000, 10)

ii = (784, 1)  #input image
il = (10, 1)  #imput label

w1 = (784, 30)
b1 = (30, 1)

w2 = (30, 20)
b2 = (20, 1)

w3 = (20, 10)
b3 = (10, 1)

"""

def relu_derivative(x):
    return (x > 0).astype(float)

images, labels = get_mnist()
# w1 = np.random.uniform(-0.5, 0.5, (30, 784))
w1 = np.random.randn(30, 784) * np.sqrt(1/784)
b1 = np.random.uniform(-0.5, 0.5, (30, 1))

# w2 = np.random.uniform(-0.5, 0.5, (20, 30))
w2 = np.random.randn(20, 30) * np.sqrt(1/30)
b2 = np.random.uniform(-0.5, 0.5, (20, 1))

# w3 = np.random.uniform(-0.5, 0.5, (10, 20))
w3 = np.random.randn(10, 20) * np.sqrt(1/20)
b3 = np.random.uniform(-0.5, 0.5, (10, 1))

learn_rate = 0.01
nr_correct = 0
epochs = 3

for epoch in range(epochs):
    for img, l in zip(images, labels):
        img = img.reshape(784, 1)
        l = l.reshape(10, 1)

        a1_p = w1 @ img + b1
        # a1 = 1 / (1 + np.exp(-a1_p)) #(30,1)
        a1 = np.maximum(0, a1_p)  # ReLU

        a2_p = w2 @ a1 + b2
        # a2 = 1 / (1 + np.exp(-a2_p)) #(20,1)
        a2 = np.maximum(0, a2_p)  # ReLU

        a3_p = w3 @ a2 + b3
        a3 = 1 / (1 + np.exp(-a3_p)) #(10,1)
        # a3 = np.maximum(0, a3_p)  # ReLU

        # error calc
        e = 1 /  len(a3) * np.sum((a3 - l)**2, axis=0)
        nr_correct += int(np.argmax(a3) == np.argmax(l))

        #backpropagation a3
        delta3 = (a3 - l) * (a3 * (1 - a3))  #(10,1)

        #backpropagation a2
        # delta2 = (w3.T @ delta3) * (a2 * (1 - a2)) #(20, 1)
        delta2 = (w3.T @ delta3) * relu_derivative(a2_p)

        #backpropagation a1
        # delta1 = (w2.T @ delta2) * (a1 * (1 - a1)) #(30, 1)
        delta1 = (w2.T @ delta2) * relu_derivative(a1_p)

        #weights update
        w3 = w3 -learn_rate * delta3 @ a2.T #(20, 1)
        b3 = b3 -learn_rate * delta3
        w2 = w2 -learn_rate * delta2 @ a1.T #(30, 1)
        b2 = b2 -learn_rate * delta2
        w1 = w1 -learn_rate * delta1 @ img.T #(784, 30)
        b1 = b1 -learn_rate * delta1
    
    print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0


