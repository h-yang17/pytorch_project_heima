import numpy as np
import torch
# torch tensor 和 numpy array共享底层的存储空间，因此改变其中的一个值，另一个值也随之改变
# 返回一个全为1 的张量，形状由可变参数sizes定义。
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)
print("++++++++++++++++++++++++++++++++++++")
# 对其中一个进行加法运算，另一个也随之改变
a.add_(1)
print(a)
print(b)
print("++++++++++++++++++++++++++++++++++++")
# 将numpy转变为tensor
a = np.ones(5)
b = torch.from_numpy(a)
# a = a + 1
np.add(a, 1, out=a)
print(a)
print(b)