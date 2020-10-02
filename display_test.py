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

def visualize_traindata():
    """
    データをどのように斟酌したか可視化
    各データの位置・ラベル(改変済み)・損失を同時にプロット

    Attribute
    x, y: データ(訓練データ等), 正解ラベル(改変済み)
    loss: 各データの損失

    """
    dataset_path = "data/dataset/multi_1000/"
    x, y, _, _, _ = dataset.load_artifical_dataset(dataset_path)

    n = len(x)

    index_0 = np.where(y == 0)
    index_1 = np.where(y == 1)
    index_2 = np.where(y == 2)
    index_3 = np.where(y == 3)

    # (改変前)label 0
    plt.scatter(x[index_0, 0], x[index_0, 1], marker='o', color='r', zorder=2)
    # (改変前)label 1
    plt.scatter(x[index_1, 0], x[index_1, 1], marker='o', color='b', zorder=2)
    # (改変前)label 2
    plt.scatter(x[index_2, 0], x[index_2, 1], marker='o', color='orange', zorder=2)
    # (改変前)label 3
    plt.scatter(x[index_3, 0], x[index_3, 1], marker='o', color='purple', zorder=2)

    plt.show()
    

# x_train, y_train, dataloader_train, dataloader_test, train_dataset = dataset.MNIST_load()
# k_index = [i for i in range(1000)]
# tSNE(x_train, y_train, k_index)
visualize_traindata()
