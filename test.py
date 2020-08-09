import matplotlib.pyplot as plt
import numpy as np

def Random(a, b):
    """ aからbまでの一様乱数を返す """
    return (b - a) * np.random.rand() + a

x = np.array([[Random(2, 4) for i in range(100)] for j in range(50)])
print(x)

plt.imshow(x.T, cmap='jet')
plt.xlabel("epoch")
plt.ylabel("data")
plt.show()