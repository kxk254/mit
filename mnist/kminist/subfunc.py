import numpy as np
import os
import random

def data_loader():
    random.seed(66)
    with np.load("../data/mnist.npz", "r") as f:
        images, labels = f["x_train"], f["y_train"]
        images_test, labels_test = f["x_test"], f["y_test"]
        images = images.astype("float32")
        images_test = images_test.astype("float32")
    
    n1 = int(images_test.shape[0] * 0.5) 
    images_val = images_test[:n1]
    labels_val = labels_test[:n1]
    images_test = images_test[n1:]
    labels_test = labels_test[n1:]
    return images, labels, images_val, labels_val, images_test, labels_test


with np.load("../data/mnist.npz", "r") as f:
        images, labels = f["x_train"], f["y_train"]
        images_test, labels_test = f["x_test"], f["y_test"]
        images = images.astype("float32")
        images_test = images_test.astype("float32")
    
n1 = int(images_test.shape[0] * 0.5) 
images_val = images_test[:n1]
labels_val = labels_test[:n1]
images_test = images_test[n1:]
labels_test = labels_test[n1:]



print(images.shape)
print(images.dtype)
print(labels.shape)
print(labels.dtype)

print(images_val.shape)
print(images_val.dtype)
print(labels_val.shape)
print(labels_val.dtype)

print(images_test.shape)
print(images_test.dtype)
print(labels_test.shape)
print(labels_test.dtype)