import torch

x = torch.rand(5, 3)
y = torch.rand(5, 3)
# 第一种加法
print(x+y)

# 第二种加法
print(torch.add(x, y))

# 第三种加法
# 提前设置一个空的张量
result = torch.empty(5, 3)
# 将空的张量作为加法的结果存储张量
torch.add(x, y, out=result)
print(result)

# 第四种加法 y = y + x in_place原地置换
'''
所以的in_place的操作的函数都有一个下划线的后缀
比如x.copy_(y) x.add_(y)都会直接改变x的值
'''
y.add_(x)
print(y)
