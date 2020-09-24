import dataset
import matplotlib.pyplot as plt
import numpy as np

# x_train, y_train, dataloader_train, dataloader_test = dataset.make_and_load_artifical_dataset(1000, 1)

# index_0 = np.where(y_train == 0)
# index_1 = np.where(y_train == 1)

# plt.plot(x_train[index_0, 0], x_train[index_0, 1], marker=".", color="red")
# plt.plot(x_train[index_1, 0], x_train[index_1, 1], marker=".", color="blue")
# plt.show()

####
 
# 乱数を生成
x = np.random.rand(100)
y = np.random.rand(100)
 
# 散布図を描画
plt.scatter(x, y, s=40, facecolors='none', edgecolors='r')

plt.show()