import numpy as np
import pathlib

def get_mnist():
    base_dir = pathlib.Path(__file__).resolve().parent
    # data_path = "H:/dev/mit/mnist/data/mnist.npz"
    data_path = "D:/develop/mit/mnist/data/mnist.npz"
    with np.load(data_path) as f:
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))


    labels = np.eye(10)[labels]
    return images, labels