import torch 
import torch.nn as nn
import torch.nn.functional as F

num_classes = 10

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = torch.nn.Linear(2, 64)
        self.fc2 = torch.nn.Linear(64, 2)
 
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class SimpleNet2(torch.nn.Module):
    def __init__(self):
        super(SimpleNet2, self).__init__()
        self.fc1 = torch.nn.Linear(2, 20)
        self.fc2 = torch.nn.Linear(20, 20)
        self.fc3 = torch.nn.Linear(20, 20)
        self.fc4 = torch.nn.Linear(20, 20)
        self.fc5 = torch.nn.Linear(20, 20)
        self.fc6 = torch.nn.Linear(20, 2)
 
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))

        return self.fc6(x)

# モデル定義
# class MNISTNet(torch.nn.Module):
#     def __init__(self):
#         super(MNISTNet, self).__init__()
#         self.fc1 = torch.nn.Linear(28*28, 1000)
#         self.fc2 = torch.nn.Linear(1000, 1000)
#         self.fc3 = torch.nn.Linear(1000, 100)
#         self.fc4 = torch.nn.Linear(100, 100)
#         self.fc5 = torch.nn.Linear(100, 100)
#         self.fc6 = torch.nn.Linear(100, 2)
 
#     def forward(self, x):
#         # テンソルのリサイズ: (N, 1, 28, 28) -> (N, 784)
#         x = x.view(-1, 28 * 28)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = F.relu(self.fc5(x))
#         return self.fc6(x)

# モデル定義
class MNISTNet(torch.nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 1000)
        self.fc2 = torch.nn.Linear(1000, 100)
        self.fc3 = torch.nn.Linear(100, 100)
        self.fc4 = torch.nn.Linear(100, 2)
 
    def forward(self, x):
        # テンソルのリサイズ: (N, 1, 28, 28) -> (N, 784)
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
