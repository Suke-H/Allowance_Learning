import matplotlib.pyplot as plt
import numpy as np
import copy

import dataset

def Random(a, b):
    """ aからbまでの一様乱数を返す """
    return (b - a) * np.random.rand() + a

# pointsからpのk近傍点のindexのリストを返す
def K_neighbor(points, p, k):
    #points[i]とpointsの各点とのユークリッド距離を格納
    distances = np.sum(np.square(points - p), axis=1)

    #距離順でpointsをソートしたときのインデックスを格納
    sorted_index = np.argsort(distances)

    return sorted_index[:k]

if __name__ == "__main__":
    dataset_path = "data/dataset/fuzzy_data_u1_b10/"
    x_train, y_train, dataloader_train, dataloader_val, dataloader_test = dataset.load_artifical_dataset(dataset_path)

    n = len(x_train) // 10
    x_train = x_train[:20]

    loss = np.array([[Random(1, 4) for i in range(n)] for i in range(50)])

    p = np.mean(x_train, axis=0)
    sorted_indices = []
    prev_list = [i for i in range(n)]

    for i in range(n):

        # 最初はx_trainの平均点から最も遠い点を選択
        if i == 0:
            indices = K_neighbor(x_train, p, n)
            p_index = indices[n-1]

        # それ以降は前に選択された点から最も近い点を選択
        else:
            indices = K_neighbor(np.delete(x_train, sorted_indices, axis=0), p, 1)
            p_index = prev_list[indices[0]]    
            
        sorted_indices.append(p_index)

        p = x_train[p_index]
        prev_list.remove(p_index)
        print(p_index)
        
        print(p.shape, len(prev_list))

    # 確認

    # プロット
    plt.plot(x_train[:, 0], x_train[:, 1],'o')

    for i in range(n):
        num = sorted_indices[i]
        plt.annotate(i, xy=(x_train[num, 0], x_train[num, 1]))

    plt.show()
    plt.close()

    # ソート
    loss = loss[sorted_indices]

    plt.imshow(loss.T, cmap='Reds')
    plt.xlabel("round")
    plt.ylabel("data")
    plt.show()
