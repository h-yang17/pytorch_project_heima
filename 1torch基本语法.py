'''
pytorch作为一个基于numpy的科学计算包，是numpy的替代者，向用户提供使用GPU强大功能的能力
'''
import torch
# 创建没有初始化的矩阵
x = torch.empty(5, 3)
print(x)

# 创建一个随机矩阵
x = torch.rand(5, 3)
print(x)

# 创建一个全零的矩阵并且指定元素的类型为long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 直接通过数值创建张量
x = torch.tensor([2.5, 3.5])
print(x)

# 通过已有的张量创建相同尺寸的新张量
x = x.new_ones(5, 3, dtype=float)
print(x)

# 利用randn_like方法得到相同尺寸的张量，并采用随机初始化的方法为其赋值
y = torch.randn_like(x, dtype=torch.double)
print(y)

# 利用.size()得到张量的尺寸
print(x.size())
print(y.size())

# torch.size函数本质上返回的是一个元组tuple，因此它支持元组的一切操作
a, b = y.size()
print(a)
print(b)