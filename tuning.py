import torch
import itertools
import os
import numpy as np
import pandas as pd
from time import time

from online import online_binary, online_MNIST, online_multi, online_check, online_multi_MNIST
from model import SimpleNet2, MNISTNet, SimpleNet3
from stop_algo import stop_algo
import dataset
from adversary import adversary

class algorithm_set():
    """
    各アルゴリズムのパラメータを管理するためのクラス
    """

    def __init__(self, algo_type):
        self.algo_type = algo_type

def dataset_load(data_type, dataset_path=None):
    """
    データセットの読み込み
    """

    if data_type == "artifact":
        return dataset.load_artifical_dataset(dataset_path)

    elif data_type == "mnist":
        return dataset.MNIST_load()

def write_stage_log(stage, batch_size=None, epoch=None, n_round=None, sigma=None, reset_flag=None, loss_type=None, limit_phase=None):
    """
    各ステージでログ表示
    flag関係の処理も同時に行う
    """

    params = []

    print("="*50)
    print("<stage: {}>".format(stage))
    if batch_size:
        print("batch_size: {}".format(batch_size))
    if epoch:    
        print("epoch: {}".format(epoch))
    if n_round:
        print("round: {}".format(n_round))
    if sigma:
        print("sigma: {}".format(sigma))

    if reset_flag is not None:
        if reset_flag == 0:
            reset_flag = False
        elif reset_flag == 1:
            reset_flag = True
        print("reset_flag: {}".format(reset_flag))
        params.append(reset_flag)   

    if loss_type:
        if loss_type == 1:
            loss_type = "loss1"
        elif loss_type == 2:
            loss_type = "1-p"
        elif loss_type == 3:
            loss_type = "p"
        print("loss_type: {}".format(loss_type))
        params.append(loss_type)

    if limit_phase:
        if limit_phase == 1:
            limit_phase = "train"
        elif limit_phase == 2:
            limit_phase = "val"
        print("limit_phase: {}".format(limit_phase))
        params.append(limit_phase)

    if reset_flag or loss_type or limit_phase:
        return params


def tuning(acc, Model, out_path, dataset_tuple, algo):
    """
    パラメータチューニング
    """

    # オンラインアルゴリズム 
    if algo.algo_type in ["online_binary", "online_multi"]:

        # パラメータをcsvで出力
        tune_list = np.array(list(itertools.product(algo.batch_size_list, algo.epoch_list, algo.round_list, algo.sigma_list, algo.reset_list, algo.loss_list)))
        df = pd.DataFrame(data=tune_list, columns=['batch_size', 'epoch', 'round', 'sigma', 'reset', 'loss'])
        df.to_csv(out_path+"para.txt")

        for i, (batch_size, epoch, n_round, sigma, reset_flag, loss_type) in enumerate(tune_list, 1):

            # 各ステージのパラメータをprint出力
            reset_flag, loss_type = write_stage_log(stage=i, batch_size=batch_size, epoch=epoch, n_round=n_round, sigma=sigma, 
                                                    reset_flag=reset_flag, loss_type=loss_type)

            os.makedirs(out_path + str(i) + "/p", exist_ok=True)
            os.makedirs(out_path + str(i) + "/d", exist_ok=True)
            os.makedirs(out_path + str(i) + "/l", exist_ok=True)

            # 2クラス分類
            if algo.algo_type == "online_binary":

                # 実行
                online_binary(acc, Model, dataset_tuple, out_path + str(i) + "/", i, 
                        batch_size=int(batch_size), train_epoch=int(epoch),  # 分類器のパラメータ
                        n_round=int(n_round), sigma=sigma,  # オンライン予測のパラメータ
                        reset_flag=reset_flag, loss_type=loss_type # 学習リセットするか、損失の種類
                        )

            # 多クラス分類
            elif algo.algo_type == "online_multi":

                # 実行
                online_multi(acc, Model, dataset_tuple, out_path + str(i) + "/", i, 
                    batch_size=int(batch_size), etrain_poch=int(epoch),  # 分類器のパラメータ
                    n_round=int(n_round), sigma=sigma,  # オンライン予測のパラメータ
                    reset_flag=reset_flag, loss_type=loss_type # 学習リセットするか、損失の種類
                    )

    # ナイーブアルゴリズム
    elif algo.algo_type == "naive":

        # パラメータをcsvで出力
        tune_list = np.array(list(itertools.product(algo.batch_size_list, algo.epoch_list, algo.limit_phase_list, algo.loss_list)))
        df = pd.DataFrame(data=tune_list, columns=['batch_size', 'epoch', 'limit_phase', 'loss'])
        df.to_csv(out_path+"para.txt")

        for i, (batch_size, epoch, limit_phase, loss_type) in enumerate(tune_list, 1):

            loss_type, limit_phase = write_stage_log(stage=i, batch_size=batch_size, epoch=epoch, 
                                                    limit_phase=limit_phase, loss_type=loss_type)

            os.makedirs(out_path + str(i) + "/p", exist_ok=True)
            os.makedirs(out_path + str(i) + "/d", exist_ok=True)
            os.makedirs(out_path + str(i) + "/l", exist_ok=True)

            # 実行
            stop_algo(acc, Model, dataset_tuple, out_path + str(i) + "/", i,
                num_epochs=int(epoch), batch_size=int(batch_size), limit_phase=limit_phase, loss_type=loss_type)

    # 敵対アルゴリズム 
    elif algo.algo_type == "adversary":

        # パラメータをcsvで出力
        tune_list = np.array(list(itertools.product(algo.batch_size_list, algo.epoch_list)))
        df = pd.DataFrame(data=tune_list, columns=['batch_size', 'epoch'])
        df.to_csv(out_path+"para.txt")

        for i, (batch_size, epoch) in enumerate(tune_list, 1):

            # 各ステージのパラメータをprint出力
            write_stage_log(stage=i, batch_size=batch_size, epoch=epoch)

            os.makedirs(out_path + str(i) + "/p", exist_ok=True)
            os.makedirs(out_path + str(i) + "/d", exist_ok=True)
            os.makedirs(out_path + str(i) + "/l", exist_ok=True)

            # 実行
            adversary(acc, Model, dataset_tuple, out_path + str(i) + "/", i, 
                    batch_size=int(batch_size), train_epoch=int(epoch),  # 分類器のパラメータ
                    )
    
if __name__ == '__main__':

    # 共通設定
    acc = 0.8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Model = SimpleNet2().to(device)
    out_path = "data/result/tuning/bag_check/test4/"

    # データセットの設定
    # artifact, mnist
    dataset_tuple = dataset_load(data_type="artifact", dataset_path="data/dataset/fuzzy_1000_1/")
    
    # アルゴリズムとパラメータの設定

    # online_binary, online_multi
    # algo = algorithm_set("online_binary")
    # algo.batch_size_list = [200]
    # algo.epoch_list = [10]
    # algo.round_list = [50]
    # algo.sigma_list = [10**(-4)]
    # # 0ならFalse, 1ならTrue
    # algo.reset_list = [0]
    # # 1: loss1, 2: 1-p, 3: p
    # algo.loss_list = [2, 2, 3, 3]

    # naive
    # algo = algorithm_set("naive")
    # algo.batch_size_list = [200]
    # algo.epoch_list = [50]
    # # 1: train, 2: val
    # algo.limit_phase_list = [1, 2]
    # # 1: loss1, 2: 1-p, 3: p
    # algo.loss_list = [1, 1, 2, 2, 3, 3]

    # adversary
    algo = algorithm_set("adversary")
    algo.batch_size_list = [200]
    algo.epoch_list = [50]
    
    # 開始
    tuning2(acc, Model, out_path, dataset_tuple, algo)



    