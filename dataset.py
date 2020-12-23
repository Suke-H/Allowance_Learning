import numpy as np
import copy    
from matplotlib import pyplot as plt
import seaborn as sns

import tensorflow as tf
import torch
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

def Random(a, b):
    """ aからbまでの一様乱数を返す """
    return (b - a) * np.random.rand() + a

def make_artificial_data(n, phase="train"):
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

def fuzzy_dataset(n, mu):
    # 平均(label0と1で変える)
    mu0 = [mu, mu]
    mu1 = [-mu, -mu]

    # 分散(label0と1で共通)
    sigma = [[1, 0], [0, 1]]
    
    # 2次元正規乱数によりデータ生成
    data0 = np.random.multivariate_normal(mu0, sigma, int(n // 2))
    data1 = np.random.multivariate_normal(mu1, sigma, int(n // 2))
    datas = np.concatenate([data0, data1], axis=0).astype(np.float32)

    # ラベル
    labels = np.array([int(i/int(n/2)) for i in range(n)])

    # シャッフル
    perm = np.random.permutation(n)
    datas, labels = datas[perm], labels[perm]

    return datas, labels

def make_artificial_dataset(path, data_fanc):
    # データセット作成
    # train_x, train_t = data_fanc(1000, 1)
    # val_x, val_t = data_fanc(1000, 1)
    # test_x, test_t = data_fanc(1000, 1)
    train_x, train_t = data_fanc(1000, 2)
    val_x, val_t = data_fanc(1000, 2)
    test_x, test_t = data_fanc(1000, 2)
    # train_x, train_t = data_fanc(1000)
    # val_x, val_t = data_fanc(1000)
    # test_x, test_t = data_fanc(1000)

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
    ds_train = data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_t))
    dataloader_train = data.DataLoader(dataset=ds_train, batch_size=200, shuffle=False)

    ds_val = data.TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_t))
    dataloader_val = data.DataLoader(dataset=ds_val, batch_size=200, shuffle=False)

    ds_test = data.TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_t))
    dataloader_test = data.DataLoader(dataset=ds_test, batch_size=200, shuffle=True)

    # return dataloader_train, dataloader_val, dataloader_test
    return train_x, train_t, dataloader_train, dataloader_test, ds_train

def make_and_load_artifical_dataset(n, mu):
    # データセット作成
    train_x, train_t = fuzzy_dataset(n, mu)
    val_x, val_t = fuzzy_dataset(n, mu)
    test_x, test_t = fuzzy_dataset(n, mu)

    # numpy -> Dataset -> DataLoader
    ds_prob = data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_t))
    dataloader_train = data.DataLoader(dataset=ds_prob, batch_size=10, shuffle=False)

    ds_val = data.TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_t))
    dataloader_val = data.DataLoader(dataset=ds_val, batch_size=100, shuffle=False)

    ds_test = data.TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_t))
    dataloader_test = data.DataLoader(dataset=ds_test, batch_size=100, shuffle=True)

    # return train_x, train_t, dataloader_train, dataloader_val, dataloader_test
    return train_x, train_t, dataloader_train, dataloader_test

class ImageTransform():
    """
    画像の前処理クラス。
    torchテンソル化と標準化を行う。
    """

    def __init__(self):
        self.data_transform = transforms.Compose(
                            [transforms.ToTensor(),
                            transforms.Normalize((0.5, ), (0.5, ))])

    def __call__(self, img):
        return self.data_transform(img)

def MNIST_load():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 前処理用の関数
    transform = ImageTransform()
    img_transformed = transform

    # データセット読み込み + 前処理
    train_dataset = torchvision.datasets.MNIST(root='./data', 
                        train=True, download=True, transform=img_transformed)
    test_dataset = torchvision.datasets.MNIST(root='./data', 
                    train=False, download=True, transform=img_transformed)

    # データセット読み込み + 前処理
    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                 batch_size=100, shuffle=False, num_workers=2)
    # val_loader = torch.utils.data.DataLoader(val_dataset, 
    #                 batch_size=100, shuffle=False, num_workers=2)                                         
    # test_loader = torch.utils.data.DataLoader(test_dataset, 
    #                 batch_size=100, shuffle=False, num_workers=2)

    # numpyに変換
    train_x = np.array([train_dataset[i][0].cpu().numpy() for i in range(60000)])
    train_y = np.array([train_dataset[i][1] for i in range(60000)])
    test_x = np.array([test_dataset[i][0].cpu().numpy() for i in range(10000)])
    test_y = np.array([test_dataset[i][1] for i in range(10000)])

    # 0と1の画像だけにする
    # train_indices = np.where((train_y == 7) | (train_y == 1))
    # train_x, train_y = train_x[train_indices], train_y[train_indices]
    # train_y = np.where(train_y == 7, 0, train_y)

    # test_indices = np.where((test_y == 7) | (test_y == 1))
    # test_x, test_y = test_x[test_indices], test_y[test_indices]
    # test_y = np.where(test_y == 7, 0, test_y)

    # datasetオブジェクト作成
    train_x_tensor = torch.Tensor(train_x).to(device)
    # train_y_tensor = torch.Tensor(train_y, dtype=torch.long).to(device)
    train_y_tensor = torch.tensor(train_y, dtype=torch.long)
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_y_tensor)

    test_x_tensor = torch.Tensor(test_x).to(device)
    # test_y_tensor = torch.Tensor(test_y, dtype=torch.long).to(device)
    test_y_tensor = torch.tensor(test_y, dtype=torch.long)
    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_y_tensor)

    # dataloaderオブジェクト作成
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    return train_x, train_y, train_loader, test_loader, train_dataset

def multi_dataset(n):
    """
    -1 <= x, y <= 1での一様乱数により作られた2次元のデータを
    label 0: 第1象限(右上)
    label 1: 第2象限(左上)
    label 2: 第3象限(左下)
    label 3: 第4象限(右下)
    の4クラスにしたデータ
    """

    # 正解ラベル
    labels = np.array([int(i/int(n/4)) for i in range(n)])
    # データ
    dataset = np.zeros((n, 2))

    for i, label in enumerate(labels):

        # label 0
        if label == 0:
            dataset[i] = [Random(0, 1), Random(0, 1)]

        # label 1
        elif label == 1:
            dataset[i] = [Random(-1, 0), Random(0, 1)]

        # label 2
        elif label == 2:
            dataset[i] = [Random(-1, 0), Random(-1, 0)]

        # label 3
        else:
            dataset[i] = [Random(0, 1), Random(-1, 0)]

    # シャッフル
    perm = np.random.permutation(n)
    dataset, labels = dataset[perm], labels[perm]

    return np.array(dataset, dtype="float32"), np.array(labels, dtype="int")

def multi_fuzzy_dataset(n, mu):
    # 平均
    mu0 = [mu, mu]
    mu1 = [-mu, mu]
    mu2 = [-mu, -mu]
    mu3 = [mu, -mu]

    # 分散(共通)
    sigma = [[1, 0], [0, 1]]
    
    # 2次元正規乱数によりデータ生成
    data0 = np.random.multivariate_normal(mu0, sigma, int(n // 4))
    data1 = np.random.multivariate_normal(mu1, sigma, int(n // 4))
    data2 = np.random.multivariate_normal(mu2, sigma, int(n // 4))
    data3 = np.random.multivariate_normal(mu3, sigma, int(n // 4))
    datas = np.concatenate([data0, data1, data2, data3], axis=0).astype(np.float32)

    # ラベル
    labels = np.array([int(i/int(n/4)) for i in range(n)])

    # シャッフル
    perm = np.random.permutation(n)
    datas, labels = datas[perm], labels[perm]

    return datas, labels

if __name__ == '__main__':
    # make_artificial_dataset("data/dataset/fuzzy_1000_1/", fuzzy_dataset)
    # make_artificial_dataset("data/dataset/multi_1000/", multi_dataset)
    # make_artificial_dataset("data/dataset/multi_fuzzy_1000_2/", multi_fuzzy_dataset)

    dataset_set = dataset.MNIST_load()