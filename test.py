import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# grid_num = 10
# aabb = [-1, 1, 0, 1]
 
# x = np.linspace(aabb[0], aabb[1], grid_num)
# y = np.linspace(aabb[2], aabb[3], grid_num)

# xx, yy = np.meshgrid(x,y)

# xy = np.stack([xx, yy])
# xy = xy.T.reshape(grid_num**2, 2)

# print(xy)

n = 200

# 平均(label0と1で変える)
mu0 = [1, 1]
mu1 = [-1, -1]
# mu0 = [3/4, 3/4]
# mu1 = [-3/4, -3/4]

# 分散(label0と1で共通
sigma = [[1, 0], [0, 1]]

# 2次元正規乱数によりデータ生成
data0 = np.random.multivariate_normal(mu0, sigma, int(n // 2))
data1 = np.random.multivariate_normal(mu1, sigma, int(n // 2))

plt.plot(data0[:, 0],data0[:, 1],marker=".",linestyle="None",color="red", label=0)
plt.plot(data1[:, 0],data1[:, 1],marker=".",linestyle="None",color="blue", label=1)
plt.plot([mu0[0], mu1[0]], [mu0[1], mu1[1]],marker="o",linestyle="None",color="black", label="μ")

plt.show()

