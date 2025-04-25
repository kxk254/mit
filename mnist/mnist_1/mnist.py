import numpy as np
from data import get_mnist

from dense import Dense
from activations import Tanh
from losses import mse, mse_prime
from network import train, predict

def to_categorical_numpy(y, num_classes):
    return np.eye(num_classes)[y]

def preprocess(x, y):
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    y = to_categorical_numpy(y, num_classes=10)
    y = y.reshape(y.shape[0], 10, 1)
    return x, y

def split_data(x, y, train_size=1000, test_size=20, seed=66):
    np.random.seed(seed)
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    x = x[indices]
    y = y[indices]

    x_train, y_train = x[:train_size], y[:train_size]
    x_test, y_test = x[train_size:train_size + test_size], y[train_size:train_size + test_size]
    return x_train, y_train, x_test, y_test

# load MIST from server
x_data, y_data = get_mnist()
x_data, y_data = preprocess(x_data, y_data)
x_train, y_train, x_test, y_test = split_data(x_data, y_data, train_size=1000, test_size=20)


# neural network
network = [
    Dense(28*28, 40),
    Tanh(),
    Dense(40, 10),
    Tanh(),
]


# train
train(network, mse, mse_prime, x_train, y_train, epochs=10, learning_rate=0.001)

# test
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print('pred', np.argmax(output), "\\ttrue:", np.argmax(y))