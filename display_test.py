import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import dataset

def tSNE(x, t, k_index):

    n = len(x)

    # reshape
    x = x.reshape(n, 28*28)

    # 改変ラベルを1000個選択
    k_index = np.random.choice(k_index, 1000, replace=False)
    x_change = x[k_index]
    t_change = t[k_index]

    x_change_reduced = TSNE(n_components=2, random_state=0).fit_transform(x_change)

    # 改変ラベル以外を2000個選択
    else_index = np.delete(np.array([i for i in range(n)]), k_index)
    else_index = np.random.choice(else_index, 2000, replace=False)

    x_else = x[else_index]
    t_else = t[else_index]

    x_else_reduced = TSNE(n_components=2, random_state=0).fit_transform(x_else)

    plt.scatter(x_change_reduced[:, 0], x_change_reduced[:, 1], c=t_change, marker="x")
    plt.scatter(x_else_reduced[:, 0], x_else_reduced[:, 1], c=t_else, marker=".")
    plt.colorbar()

    plt.show()
    

# x_train, y_train, dataloader_train, dataloader_test, train_dataset = dataset.MNIST_load()
# k_index = [i for i in range(1000)]
# tSNE(x_train, y_train, k_index)
