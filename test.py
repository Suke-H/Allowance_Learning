import torch

a = torch.randn((4,4))
print(a)

b = torch.tensor([[0],[1],[2],[3]])
print(b)

print(b.reshape(1, 1, 4))

c = a.gather(1, b).squeeze()
print(c)