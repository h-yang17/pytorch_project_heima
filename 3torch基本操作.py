import torch
# 对张量进行切片
x = torch.rand(5, 3)
print(x)
# 得到第一列的所有元素
print(x[:, 1])
# 得到第0 ， 1列
print(x[:, :2])

print("++++++++++++++++++++++++++++++++++++")
# 改变张量的形状
x = torch.rand(4, 4)
# torch.view()保证数据元素总数量不变
# -1代表自动匹配
y = x.view(-1, 8)
z = x.view(16)
print(x.size(), y.size(), z.size())

print("++++++++++++++++++++++++++++++++++++")
# 如果张量只有一个元素，可以使用item取出，作为一个python number
x = torch.rand(1)
print(x)
print(x.item())