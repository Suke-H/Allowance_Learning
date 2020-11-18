import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from online import train, init_weights, softmax, softmax2
from visual import visualization_multi
import dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def cal_p_binary(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    label_RightorWrong = []
    with torch.no_grad():
        p_list = []
        for step, (images, labels) in enumerate(dataloader, 1):
            
            labelinf = np.zeros(len(labels))
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
           
            _, predicted = torch.max(outputs.data, 1)
            
            p = softmax([list(map(float,i)) for i in list(outputs)])      
            p_list.extend(p)
            
            prelist = list(map(int,predicted))
            lablist = list(map(int,labels))
            
            labelinf[np.where(np.array(prelist) == np.array(lablist))[0]] = 1
            labelinf[np.where(np.array(prelist) != np.array(lablist))[0]] = -1
            
            label_RightorWrong.extend(labelinf)

        train_acc = len(np.where(np.array(label_RightorWrong)==1)[0])/len(label_RightorWrong)
    
    return np.array(p_list), train_acc

def cal_p_multi(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    label_RightorWrong = []
    with torch.no_grad():

        p_list = []
        # new_p_list = []
        # クラス数
        p_class_list = np.empty((0, 4))
        # p_class_list = np.empty((0, 10))
        
        for step, (images, labels) in enumerate(dataloader, 1):
            
            labelinf = np.zeros(len(labels))
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
           
            _, predicted = torch.max(outputs.data, 1)

            p = softmax([list(map(float,i)) for i in list(outputs)])      
            p_list.extend(p)

            # new_p = softmax3([list(map(float,i)) for i in list(outputs)], labels.to('cpu').detach().numpy().copy())
            # new_p_list.extend(new_p)    

            p_class = softmax2([list(map(float,i)) for i in list(outputs)])

            p_class_list = np.append(p_class_list, p_class, axis=0)
            prelist = list(map(int,predicted))
            lablist = list(map(int,labels))
            
            labelinf[np.where(np.array(prelist) == np.array(lablist))[0]] = 1
            labelinf[np.where(np.array(prelist) != np.array(lablist))[0]] = -1
            
            label_RightorWrong.extend(labelinf)

        train_acc = len(np.where(np.array(label_RightorWrong)==1)[0])/len(label_RightorWrong)
    
    return np.array(p_class_list), np.array(p_list), train_acc

def eval(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    y = []
    with torch.no_grad():
        for (images, labels) in dataloader:
            images, labels = images.to(device), labels.to(device)
 
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            predicted = predicted.to('cpu').detach().numpy().copy().tolist()
            y.extend(predicted)
 
    print("Test Acc : %.4f" % (correct/total))

    return correct/total, np.array(y)

def adversary(acc, Model, dataset_tuple, out_path, tune_epoch, 
                classfy_type, 
                batch_size=10, n_epochs=10 # 分類器のパラメータ
                ):

    torch.manual_seed(1)

    # 入力データをロード
    x_train, y_train, dataloader_train, dataloader_val, dataloader_test = dataset_tuple
    # x_train, y_train, dataloader_train, dataloader_test = dataset.make_and_load_artifical_dataset(data_num, mu)

    # 入力データを可視化
    # init_visual(x_train, y_train, out_path)

    # ネットワークの重みを初期化
    Model.apply(init_weights)

    # n: データ数
    n = len(x_train)
    # k: 改変するデータ数
    k = round(n * (1-acc))
    print(n, k)

    # パラメータ
    lr = 10**(-2)
    Criterion = nn.CrossEntropyLoss()
    Optimizer = optim.Adam(Model.parameters(), lr=lr)

    train_acclist = []
    test_acclist = []

    # 学習
    for i in range(n_epochs):
        train(Model, dataloader_train, Optimizer, Criterion)

    # p(尤度), train_accを出す
    if classfy_type == "binary":
        p_list, train_acc = cal_p_binary(Model, dataloader_train)

    elif classfy_type == "multi":
        p_class_list, p_list, train_acc = cal_p_multi(Model, dataloader_train)

    # 訓練データ
    _, y_eval = eval(Model, dataloader_train)

    # 推論
    test_acc, _ = eval(Model, dataloader_test)

    print(p_list.shape)

    # pが低い上位k個を選択
    # print(np.sort(p_list))
    xt = np.argsort(p_list)[:k]

     ### 改変パート ###
    # 損失の小さいtop-kをひっくり返す
    if classfy_type == "binary":
        y_eval[xt] = (y_train[xt] + 1) % 2

    # 多クラス分類の場合、元のラベルを除いて尤度最大のクラスを改変ラベルにする
    elif classfy_type == "multi":
        p_temp_list = np.copy(p_class_list)
        p_temp_list[[i for i in range(n)], y_train] = 0
        y_eval[xt] = np.argmax(p_temp_list[xt], axis=1)

    print("train acc: {}".format(train_acc))
    print("test acc: {}".format(test_acc))

    # 可視化
    x_train = torch.from_numpy(x_train).clone()
    visualization_multi(Model, x_train, y_eval, y_train, 0, tune_epoch, "d", out_path + "d/")

    # train_acclist.append(train_acc)
    # test_acclist.append(test_acc)
    