import numpy as np
import copy    
from matplotlib import pyplot as plt
import seaborn as sns

import tensorflow as tf
import torch
import torch.utils.data as data

def Random(a, b):
    """ aからbまでの一様乱数を返す """
    return (b - a) * np.random.rand() + a
    
def noisy_label_dataset(val_ratio,noise_ratio):
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_val = np.moveaxis(x_train, [3, 1, 2], [1, 2, 3]).astype('float32')[len(x_train)-int(len(x_train)*val_ratio):len(x_train)]
    x_train = np.moveaxis(x_train, [3, 1, 2], [1, 2, 3]).astype('float32')[:len(x_train)-int(len(x_train)*val_ratio)]
    x_test = np.moveaxis(x_test, [3, 1, 2], [1, 2, 3]).astype('float32')

    x_train /= 255
    x_val /= 255
    x_test /= 255

    label_list = [0,1,2,3,4,5,6,7,8,9]
    
    y_val = y_train.reshape(-1).astype('long')[len(x_train):]
    y_val_rand = copy.copy(y_val)
    for i in range(int(len(y_val)*noise_ratio)):
        removedlist = copy.copy(label_list)
        removedlist.remove(y_val_rand[i])
        y_val_rand[i] = np.random.choice(removedlist)

    
    y_train = y_train.reshape(-1).astype('long')[:len(x_train)]
    y_train_rand = copy.copy(y_train)
    for i in range(int(len(y_train)*noise_ratio)):
        removedlist = copy.copy(label_list)
        removedlist.remove(y_train_rand[i])
        y_train_rand[i] = np.random.choice(removedlist)

    y_test = y_test.reshape(-1).astype('long')

    ds_prob = data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_rand))
    dataloader_prob = data.DataLoader(dataset=ds_prob, batch_size=1000, shuffle=False)

    ds_val = data.TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val_rand))
    dataloader_val = data.DataLoader(dataset=ds_val, batch_size=1000, shuffle=False)

    ds_test = data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    dataloader_test = data.DataLoader(dataset=ds_test, batch_size=1000, shuffle=True)
    
    return dataloader_prob, dataloader_val, dataloader_test, x_train, y_train_rand

def make_artificial_data(n, phase):
    """
    -1 <= x, y <= 1での一様乱数により作られた2次元のデータを
    label 0: 右
    label 1: 左
    の2クラスにしたデータ

    train: 各ラベルの数を同じにしたデータ生成
    val, test: ランダムにデータ生成
    """

    # trainデータ生成
    if phase == "train":
        # 正解ラベル
        labels = np.array([int(i/int(n/2)) for i in range(n)])
        # データ
        dataset = np.zeros((n, 2))

        for i, label in enumerate(labels):

            # label 0
            if label == 0:
                dataset[i] = [Random(0, 1), Random(-1, 1)]

            # label 1
            else:
                dataset[i] = [Random(-1, 0), Random(-1, 1)]

        # シャッフル
        perm = np.random.permutation(n)
        dataset, labels = dataset[perm], labels[perm]

    # val, testデータ生成
    else:
        # データ
        dataset = np.array([[Random(-1, 1), Random(-1, 1)] for i in range(n)])

        # 正解ラベル
        labels = np.zeros(n)

        for i, data in enumerate(dataset):
            x, y = data

            # label 0
            if x >= 0:
                labels[i] = 0

            # label 1
            else:
                labels[i] = 1

    return np.array(dataset, dtype="float32"), np.array(labels, dtype="int")

def assign_labels(datas):

    y = []

    for x, y in datas:
        if (-3/4 <= x <= -1/4) and (-1/4 <= y <= 1/4):
            y.append(0)
        
        elif (1/4 <= x <= 3/4) and (-1/4 <= y <= 1/4):
            y.append(1)

        elif x >= 0:
            y.append(0)

        else:
            y.append(1)

    return np.array(y)

def make_testdata(n):
    """
    -1 <= x, y <= 1での一様乱数により作られた2次元のデータを
    label 0: 右
    label 1: 左
    の2クラスにしたデータ

    """

    datas = np.array([[Random(-1, 1), Random(-1, 1)] for i in range(n)])
    labels = []

    for x, y in datas:
        # if (-3/4 <= x <= -1/4) and (-1/4 <= y <= 1/4):
        #     labels.append(0)
        
        # elif (1/4 <= x <= 3/4) and (-1/4 <= y <= 1/4):
        #     labels.append(1)

        if x <= -3/4:
            labels.append(0)
        
        elif 3/4 <= x:
            labels.append(1)

        elif x >= 0:
            labels.append(0)

        else:
            labels.append(1)

    return np.array(datas, dtype="float32"), np.array(labels, dtype="int")


def make_artificial_dataset(path):
    # データセット作成
    # train_x, train_t = make_artificial_data(1000, "train")
    # val_x, val_t = make_artificial_data(100, "val")
    # test_x, test_t = make_artificial_data(100, "test")

    train_x, train_t = make_testdata(1000)
    val_x, val_t = make_testdata(100)
    test_x, test_t = make_testdata(100)

    # データ保存
    np.save(path + "train_x", train_x)
    np.save(path + "train_t", train_t)
    np.save(path + "val_x", val_x)
    np.save(path + "val_t", val_t)
    np.save(path + "test_x", test_x)
    np.save(path + "test_t", test_t)

def load_artifical_dataset(path):
    # データ読み込み
    train_x = np.load(path + "train_x.npy")
    train_t = np.load(path + "train_t.npy")
    val_x = np.load(path + "val_x.npy")
    val_t = np.load(path + "val_t.npy")
    test_x = np.load(path + "test_x.npy")
    test_t = np.load(path + "test_t.npy")

    # numpy -> Dataset -> DataLoader
    ds_prob = data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_t))
    dataloader_train = data.DataLoader(dataset=ds_prob, batch_size=100, shuffle=False)

    ds_val = data.TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_t))
    dataloader_val = data.DataLoader(dataset=ds_val, batch_size=100, shuffle=False)

    ds_test = data.TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_t))
    dataloader_test = data.DataLoader(dataset=ds_test, batch_size=100, shuffle=True)

    # return dataloader_train, dataloader_val, dataloader_test
    return train_x, train_t, dataloader_val, dataloader_test

if __name__ == '__main__':
    make_artificial_dataset("data/test_arti2/")
