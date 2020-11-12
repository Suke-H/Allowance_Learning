import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
#import torchvision.models as models

import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time
from sklearn.manifold import TSNE

def Random(a, b):
    """ aからbまでの一様乱数を返す """
    return (b - a) * np.random.rand() + a

def K_neighbor(points, p, k):
    """ pointsからpのk近傍点のindexのリストを返す """

    #points[i]とpointsの各点とのユークリッド距離を格納
    distances = np.sum(np.square(points - p), axis=1)

    #距離順でpointsをソートしたときのインデックスを格納
    sorted_index = np.argsort(distances)

    return sorted_index[:k]

def acc_plot(train_accs_ori, train_accs_change, test_accs, goal_acc, root_path, file_no):
    """ 学習記録(epoch-acc)をプロット """

    end_epoch = len(train_accs_ori)

    plt.figure()
    runs = [i for i in range(1, end_epoch+1)]
    plt.plot(runs, train_accs_ori, label='train_acc_ori')
    plt.plot(runs, train_accs_change, label='train_acc_change')
    plt.plot(runs, test_accs, label='test_acc')
    plt.plot(runs, [goal_acc for i in range(end_epoch)], label='goal_acc')
    plt.xlabel('round',fontsize=16)
    plt.ylabel('acc',fontsize=16)
    plt.xticks([i for i in range(1, end_epoch+1, 5)],fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=15)
    # plt.rc('legend', fontsize=20)
    plt.savefig(root_path+"acc_"+str(file_no)+".png", bbox_inches='tight')
    plt.close()

def acc_plot2(train_accs_ori, train_accs_change, test_accs, check_accs, goal_acc, root_path, file_no):
    """ 学習記録(epoch-acc)をプロット """

    end_epoch = len(train_accs_ori)

    plt.figure()
    runs = [i for i in range(1, end_epoch+1)]
    plt.plot(runs, train_accs_ori, label='train_acc_ori')
    plt.plot(runs, train_accs_change, label='train_acc_change')
    plt.plot(runs, test_accs, label='test_acc')
    plt.plot(runs, check_accs, label='check_accs')
    plt.plot(runs, [goal_acc for i in range(end_epoch)], label='goal_acc')
    plt.xlabel('round',fontsize=16)
    plt.ylabel('acc',fontsize=16)
    plt.xticks([i for i in range(1, end_epoch+1, 5)],fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=15)
    # plt.rc('legend', fontsize=20)
    plt.savefig(root_path+"acc_"+str(file_no)+".png", bbox_inches='tight')
    plt.close()

def visualize_classify(ax1, net, aabb, grid_num=100):
    """
    識別関数を可視化

    Attribute
    grid_num: AABBの範囲でグリットを刻む回数
    aabb: AABB(Axis Aligned Bounding Box)の座標[xmin, xmax, ymin, ymax]

    """
    ### 格子点を入力する準備
    x = np.linspace(aabb[0], aabb[1], grid_num)
    y = np.linspace(aabb[2], aabb[3], grid_num)
    xx, yy = np.meshgrid(x,y)
    xy = np.stack([xx, yy])
    xy = xy.T.reshape(grid_num**2, 2)

    # 入力データ
    grid_x = np.array(xy, dtype="float32")
    grid_x = torch.from_numpy(grid_x)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_x = grid_x.to(device)

    # 格子点を入力にする
    grid_output = net(grid_x)
    # ラベルを予測
    _, grid_y = torch.max(grid_output, 1)

    grid_y = grid_y.to('cpu').detach().numpy().copy()

    ### 識別関数 可視化
    # グリッドをクラス分け
    index0 = np.where(grid_y == 0)
    index1 = np.where(grid_y == 1)
    label0 = "0"
    label1 = "1"

    grid_x = grid_x.to('cpu').detach().numpy().copy()
    grid0, grid1 = grid_x[index0], grid_x[index1]

    ax1.plot(grid0[:, 0], grid0[:, 1],marker=".",linestyle="None",color="palevioletred", label=label0, zorder=1)
    ax1.plot(grid1[:, 0], grid1[:, 1],marker=".",linestyle="None",color="lightsteelblue", label=label1, zorder=1)

    # # 凡例の表示
    # ax1.legend(markerscale=3, fontsize=16)

def visualize_multi_classify(ax1, net, aabb, grid_num=100):
    """
    識別関数を可視化

    Attribute
    grid_num: AABBの範囲でグリットを刻む回数
    aabb: AABB(Axis Aligned Bounding Box)の座標[xmin, xmax, ymin, ymax]

    """
    ### 格子点を入力する準備
    x = np.linspace(aabb[0], aabb[1], grid_num)
    y = np.linspace(aabb[2], aabb[3], grid_num)
    xx, yy = np.meshgrid(x,y)
    xy = np.stack([xx, yy])
    xy = xy.T.reshape(grid_num**2, 2)

    # 入力データ
    grid_x = np.array(xy, dtype="float32")
    grid_x = torch.from_numpy(grid_x)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_x = grid_x.to(device)

    # 格子点を入力にする
    grid_output = net(grid_x)
    # ラベルを予測
    _, grid_y = torch.max(grid_output.data, 1)
    grid_y = grid_y.to('cpu').detach().numpy().copy()

    ### 識別関数 可視化
    # グリッドをクラス分け
    index0 = np.where(grid_y == 0)
    index1 = np.where(grid_y == 1)
    index2 = np.where(grid_y == 2)
    index3 = np.where(grid_y == 3)
    label0 = "0"
    label1 = "1"
    label2 = "2"
    label3 = "3"

    grid_x = grid_x.to('cpu').detach().numpy().copy()
    grid0, grid1 = grid_x[index0], grid_x[index1]

    grid0, grid1, grid2, grid3 = grid_x[index0], grid_x[index1], grid_x[index2], grid_x[index3]

    ax1.plot(grid0[:, 0], grid0[:, 1],marker=".",linestyle="None",color="palevioletred", label=label0, zorder=1)
    ax1.plot(grid1[:, 0], grid1[:, 1],marker=".",linestyle="None",color="lightsteelblue", label=label1, zorder=1)
    ax1.plot(grid2[:, 0], grid2[:, 1],marker=".",linestyle="None",color="navajowhite", label=label2, zorder=1)
    ax1.plot(grid3[:, 0], grid3[:, 1],marker=".",linestyle="None",color="orchid", label=label3, zorder=1)

    # # 凡例の表示
    # ax1.legend(markerscale=3, fontsize=16)

def visualize_allowance(ax1, ax2, x, y, true_y, loss, epoch, vis_type, color_step=100, cmap_type="jet"):
    """
    データをどのように斟酌したか可視化
    各データの位置・ラベル(改変済み)・損失を同時にプロット

    Attribute
    x, y: データ(訓練データ等), 正解ラベル(改変済み)
    loss: 各データの損失

    """
    n = len(x)

    # color-map
    cm = plt.get_cmap(cmap_type, color_step)

    # lossを正規化(min ~ max -> 0 ~ 1)
    loss_max, loss_min = np.max(loss), np.min(loss)

    if loss_min != loss_max:
        loss = (loss - loss_min) / (loss_max - loss_min)

    # lossを(0, 1, ..., color_step-1)の離散値に変換
    loss_digit = (loss // (1 / color_step)).astype(np.int)

    # loss_digitの値がcolor_stepのものがあればcolor_step-1に置き換え
    loss_digit = np.where(loss_digit == color_step, color_step-1, loss_digit)

    # 
    index_0 = np.where(true_y == 0)
    index_1 = np.where(true_y == 1)

    # (改変前)label 0
    ax1.scatter(x[index_0, 0], x[index_0, 1], marker='o', color=cm(loss[index_0]), zorder=2)
    # (改変前)label 1
    ax1.scatter(x[index_1, 0], x[index_1, 1], marker='x', color=cm(loss[index_1]), zorder=2)

    # ax1.set_title("round: {}".format(epoch))

    # 凡例の表示
    ax1.legend(markerscale=3, fontsize=14)

    # color-barを表示
    gradient = np.linspace(0, 1, cm.N)
    gradient_array = np.vstack((gradient, gradient))
    
    ax2.imshow(gradient_array, aspect='auto', cmap=cm)
    ax2.set_axis_off()

    ax2.set_title("cumulative loss at {}th round".format(epoch), fontsize=18)

def visualize_allowance_simple(ax1, ax2, x, y, true_y, loss, epoch, vis_type, color_step=100, cmap_type="jet"):
    """
    データをどのように斟酌したか可視化
    各データの位置・ラベル(改変済み)・損失を同時にプロット

    Attribute
    x, y: データ(訓練データ等), 正解ラベル(改変済み)
    loss: 各データの損失

    """
    n = len(x)

    # color-map
    cm = plt.get_cmap(cmap_type, color_step)

    # 場合分け
    index_0_kaihen = np.where((true_y == 0) & (true_y ^ y == 1))
    index_1_kaihen = np.where((true_y == 1) & (true_y ^ y == 1))
    index_0 = np.where((true_y == 0) & (true_y ^ y == 0))
    index_1 = np.where((true_y == 1) & (true_y ^ y == 0))

    # (改変前)label 0
    ax1.scatter(x[index_0, 0], x[index_0, 1], marker='o', color='r', zorder=2)
    # (改変前)label 1
    ax1.scatter(x[index_1, 0], x[index_1, 1], marker='o', color='b', zorder=2)
    # (改変前)label 0
    ax1.scatter(x[index_0_kaihen, 0], x[index_0_kaihen, 1], s=40, facecolors='none', edgecolors='r', zorder=2)
    # (改変前)label 1
    ax1.scatter(x[index_1_kaihen, 0], x[index_1_kaihen, 1], s=40, facecolors='none', edgecolors='b', zorder=2)

    # ax1.set_title("round: {}".format(epoch))

    # 凡例の表示
    # ax1.legend(markerscale=3, fontsize=14)

    # color-barを表示
    # gradient = np.linspace(0, 1, cm.N)
    # gradient_array = np.vstack((gradient, gradient))
    
    # ax2.imshow(gradient_array, aspect='auto', cmap=cm)
    # ax2.set_axis_off()

    ax2.set_title("cumulative loss at {}th round".format(epoch), fontsize=18)

def visualize_allowance_simple_multi(ax1, ax2, x, y, true_y, loss, epoch, vis_type, color_step=100, cmap_type="jet"):
    """
    データをどのように斟酌したか可視化
    各データの位置・ラベル(改変済み)・損失を同時にプロット

    Attribute
    x, y: データ(訓練データ等), 正解ラベル(改変済み)
    loss: 各データの損失

    """
    n = len(x)

    # color-map
    cm = plt.get_cmap(cmap_type, color_step)

    index_0_kaihen = np.where((y == 0) & (true_y != y))
    index_1_kaihen = np.where((y == 1) & (true_y != y))
    index_2_kaihen = np.where((y == 2) & (true_y != y))
    index_3_kaihen = np.where((y == 3) & (true_y != y))

    index_0 = np.where(true_y == 0)
    index_1 = np.where(true_y == 1)
    index_2 = np.where(true_y == 2)
    index_3 = np.where(true_y == 3)
    
    # (改変前)label 0
    ax1.scatter(x[index_0, 0], x[index_0, 1], marker='o', color='r', zorder=2)
    # (改変前)label 1
    ax1.scatter(x[index_1, 0], x[index_1, 1], marker='o', color='b', zorder=2)
    # (改変前)label 2
    ax1.scatter(x[index_2, 0], x[index_2, 1], marker='o', color='orange', zorder=2)
    # (改変前)label 3
    ax1.scatter(x[index_3, 0], x[index_3, 1], marker='o', color='purple', zorder=2)

    # (改変前)label 0
    ax1.scatter(x[index_0_kaihen, 0], x[index_0_kaihen, 1], s=60, facecolors='none', edgecolors='r', zorder=3)
    # (改変前)label 1
    ax1.scatter(x[index_1_kaihen, 0], x[index_1_kaihen, 1], s=60, facecolors='none', edgecolors='b', zorder=3)
    # (改変前)label 2
    ax1.scatter(x[index_2_kaihen, 0], x[index_2_kaihen, 1], s=60, facecolors='none', edgecolors='orange', zorder=3)
    # (改変前)label 3
    ax1.scatter(x[index_3_kaihen, 0], x[index_3_kaihen, 1], s=60, facecolors='none', edgecolors='purple', zorder=3)

    # ax1.set_title("round: {}".format(epoch))

    # 凡例の表示
    # ax1.legend(markerscale=3, fontsize=14)

    # color-barを表示
    # gradient = np.linspace(0, 1, cm.N)
    # gradient_array = np.vstack((gradient, gradient))
    
    # ax2.imshow(gradient_array, aspect='auto', cmap=cm)
    # ax2.set_axis_off()

    ax2.set_title("cumulative loss at {}th round".format(epoch), fontsize=18)

def visualize_weights(x_train, loss, round, path):
    """
    各ラウンドにおける各訓練データの損失の遷移を可視化

    各データは

    1. 全データの平均となる点pを求める
    2. p から一番離れている点（ユークリッド距離が一番大きい点）を 次の p とする
    3. p から（ユークリッド距離で）近い点を 次の p とする
    4. 3. を繰り返して、p に選択された順にデータを並べる

    というように上から並べる

    Attribute
    x_train: 訓練データ
    loss: 各データの損失
    path: 画像の保存先

    """

    n = len(x_train)

    # reshape
    # x_train = x_train.reshape(n, 28*28)

    # p: 選択された点
    # 最初はx_trainの平均点で初期化
    p = np.mean(x_train, axis=0)

    # sorted_indices: pの選択順に訓練データのインデックスを並べたリスト
    sorted_indices = np.array([])
    # remain_list: 選択されていない点のリスト
    remain_list = [i for i in range(n)]

    # 最初はx_trainの平均点から最も遠い点を選択
    indices = K_neighbor(x_train, p, n)
    p_index = indices[n-1]

    # p に選択されたデータのインデックスを保存
    sorted_indices = np.append(sorted_indices, p_index)
    # 次の p
    p = x_train[p_index]
    # remain_listから p を削除
    remain_list.remove(p_index)

    # p に近い順から選択していく
    indices = K_neighbor(np.delete(x_train, sorted_indices, axis=0), p, n-1)
    indices = np.array(remain_list)[indices] 

    # p に選択されたデータのインデックスを保存
    sorted_indices = np.concatenate([sorted_indices, indices])

    # 結果からlossを並び替え
    sorted_indices = sorted_indices.astype(int)
    loss = loss[:, sorted_indices]

    # # データの番号を確認する用のプロット
    # plt.plot(x_train[:, 0], x_train[:, 1], 'o')
    # for i in range(n):
    #     num = sorted_indices[i]
    #     plt.annotate(i, xy=(x_train[num, 0], x_train[num, 1]))
    # # plt.show()
    # plt.savefig(path + "label.png")
    # plt.close()

    round_n = loss.shape[0]

    # 重みを正規化
    loss_min, loss_max = np.min(loss, axis=1), np.max(loss, axis=1)
    loss_normed = np.array([(loss[i] - loss_min[i]) / (loss_max[i] - loss_min[i]) for i in range(round_n)])

    # 重みの可視化
    plt.imshow(loss_normed.T, cmap='Reds', aspect=0.1)
    # plt.imshow(loss_normed.T, cmap='Reds', aspect=0.005)
    plt.xlabel("rounds")
    plt.ylabel("datas")
    # plt.colorbar()
    plt.xticks([i for i in range(1, round_n+1, 5)],fontsize=12)
    # plt.show()
    plt.savefig(path + "weights" + str(round) + ".png")
    plt.close()

def init_visual(x, y, path):
    """
    データをどのように斟酌したか可視化
    各データの位置・ラベル(改変済み)・損失を同時にプロット

    Attribute
    x, y: データ(訓練データ等), 正解ラベル(改変済み)
    loss: 各データの損失

    """
    n = len(x)

    # プロット
    for i in range(n):
        plt.plot(x[i, 0], x[i, 1], 'o')
        plt.annotate(y[i], xy=(x[i, 0], x[i, 1]))
    # plt.show()
    plt.savefig(path + "init.png")
    plt.close()

def visualization(net, x, y, true_y, loss, epoch, vis_type, path):
    """
    Attribute

    net: 学習モデル
    x, y: データ(訓練データ等), 正解ラベル(改変済み)
    loss: 各データの損失

    epoch: 何エポック目での可視化か
    vis_type: lossかpかdか
    path: 保存するフォルダのパス

    """
    # グラフ作成
    fig = plt.figure()
    ax1  = fig.add_axes((0.1,0.3,0.8,0.6))
    ax2 = fig.add_axes((0.1,0.1,0.8,0.05))

    # aabb(点を覆うxy軸平行な長方形)の座標
    aabb = [np.min(x[:, 0]), np.max(x[:, 0]), np.min(x[:, 1]), np.max(x[:, 1])]

    # 識別関数の可視化
    visualize_classify(ax1, net, aabb)

    # 斟酌の可視化
    # visualize_allowance(ax1, ax2, x, y, true_y, loss, epoch, vis_type)
    visualize_allowance_simple(ax1, ax2, x, y, true_y, loss, epoch, vis_type)

    # plt.show()
    plt.savefig(path + str(epoch) + ".png")
    plt.close()

def visualization_multi(net, x, y, true_y, loss, epoch, vis_type, path):
    """
    Attribute

    net: 学習モデル
    x, y: データ(訓練データ等), 正解ラベル(改変済み)
    loss: 各データの損失

    epoch: 何エポック目での可視化か
    vis_type: lossかpかdか
    path: 保存するフォルダのパス

    """
    # グラフ作成
    fig = plt.figure()
    ax1  = fig.add_axes((0.1,0.3,0.8,0.6))
    ax2 = fig.add_axes((0.1,0.1,0.8,0.05))

    # aabb(点を覆うxy軸平行な長方形)の座標
    aabb = [np.min(x[:, 0]), np.max(x[:, 0]), np.min(x[:, 1]), np.max(x[:, 1])]

    # 識別関数の可視化
    visualize_multi_classify(ax1, net, aabb)

    # 斟酌の可視化
    # visualize_allowance(ax1, ax2, x, y, true_y, loss, epoch, vis_type)
    visualize_allowance_simple_multi(ax1, ax2, x, y, true_y, loss, epoch, vis_type)

    # plt.show()
    plt.savefig(path + str(epoch) + ".png")
    plt.close()

def display30(dataset, indices, tune_epoch, out_path):

    #表示領域を設定（行，列）
    fig, ax = plt.subplots(5, 6)  

    #図を配置
    for i, (idx) in enumerate(indices):
        plt.subplot(5,6,i+1)
        title = str(idx) + "(" + str(dataset[idx][1].item()) + ")"
        plt.title(title, fontsize=10)    #タイトルを付ける
        plt.tick_params(color='white')      #メモリを消す
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        plt.imshow(dataset[idx][0].to('cpu').detach().numpy().copy().reshape(28,28), cmap="gray")   #図を入れ込む

    #図が重ならないようにする
    plt.tight_layout()

    #保存
    plt.savefig(out_path + "display_" + str(tune_epoch) + ".png")
    plt.close()

def tSNE(x, t, k_index, tune_epoch, out_path):

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

    #保存
    plt.savefig(out_path + "tSNE_" + str(tune_epoch) + ".png")
    plt.close()

def and_index(index_1, index_2):
    return list(filter(lambda x: x in index_2, index_1))

def tSNE2(x, t, k_index, tune_epoch, out_path):
    """
    各ラベルごとにtSNE
    """

    # for i in range(10):
    #     label_index = np.where(t == i)[0]
    #     tSNE_part_by_label(x, t, i, label_index, k_index, tune_epoch, out_path)

    label_index = np.where((t == 1) | (t == 7))[0]
    tSNE_part_by_label(x, t, 17, label_index, k_index, tune_epoch, out_path)

def tSNE_part_by_label(x, t, label, label_index, k_index, tune_epoch, out_path):

    n = len(x)

    # reshape
    x = x.reshape(n, 28*28)

    # 改変ラベル　かつ　指定したlabelの画像を抽出
    change_index = and_index(k_index, label_index)

    # 改変ラベルを1000個選択
    if len(change_index) > 1000:
        change_index = np.random.choice(change_index, 1000, replace=False)
    x_change = x[change_index]
    t_change = t[change_index]

    # tSNE
    x_change_reduced = TSNE(n_components=2, random_state=0).fit_transform(x_change)

    # 改変ラベル以外　かつ　指定したlabelの画像を抽出
    else_index = np.delete(np.array([i for i in range(n)]), k_index)
    else_index = and_index(else_index, label_index)

    # 改変ラベル以外を2000個選択
    if len(else_index) > 2000:
        else_index = np.random.choice(else_index, 2000, replace=False)
    x_else = x[else_index]
    t_else = t[else_index]

    # tSNE
    x_else_reduced = TSNE(n_components=2, random_state=0).fit_transform(x_else)

    # 描画
    plt.scatter(x_change_reduced[:, 0], x_change_reduced[:, 1], c=t_change, marker="x")
    plt.scatter(x_else_reduced[:, 0], x_else_reduced[:, 1], c=t_else, marker=".")
    plt.colorbar()

    #保存
    plt.savefig(out_path + "tSNE_" + str(tune_epoch) + "_label" + str(label) + ".png")
    plt.close()