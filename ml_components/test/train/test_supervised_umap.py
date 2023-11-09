# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np

# %matplotlib inline
import seaborn as sns
import umap
from mnist.loader import MNIST


def test_supervised_umap():
    sns.set(style="white", context="poster")
    mndata = MNIST("fashion-mnist/data/fashion")
    train, train_labels = mndata.load_training()
    test, test_labels = mndata.load_testing()
    data = np.array(np.vstack([train, test]), dtype=np.float64) / 255.0
    target = np.hstack([train_labels, test_labels])
    print("target: ", target.shape, target)
    print("data: ", data.shape)
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    embedding = umap.UMAP(n_neighbors=5).fit_transform(data, y=target)
    print("embedding: ", embedding.shape, embedding[:, 1])
