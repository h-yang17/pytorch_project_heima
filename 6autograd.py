'''
在整个Pytorch框架中, 所有的神经网络本质上都是一个autograd package(自动求导工具包)
autograd package提供了一个对Tensors上所有的操作进行自动微分的功能.
torch.Tensor是整个package中的核心类, 如果将属性.requires_grad设置为True, 它将追踪在这个类上定义的所有操作. 当代码要进行反向传播的时候, 直接调用.backward()就可以自动计算所有的梯度. 在这个Tensor上的所有梯度将被累加进属性.grad中.
如果想终止一个Tensor在计算图中的追踪回溯, 只需要执行.detach()就可以将该Tensor从计算图中撤下, 在未来的回溯计算中也不会再计算该Tensor.
除了.detach(), 如果想终止对计算图的回溯, 也就是不再进行方向传播求导数的过程, 也可以采用代码块的方式with torch.no_grad():, 这种方式非常适用于对模型进行预测的时候, 因为预测阶段不再需要对梯度进行计算.

关于torch.Function:
Function类是和Tensor类同等重要的一个核心类, 它和Tensor共同构建了一个完整的类, 每一个Tensor拥有一个.grad_fn属性, 代表引用了哪个具体的Function创建了该Tensor.
如果某个张量Tensor是用户自定义的, 则其对应的grad_fn is None
'''
import torch

x1 = torch.ones(3, 3)
print(x1)
# 如果某个张量Tensor是用户自定义的, 则其对应的grad_fn is None
x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)

print(x.grad_fn)
print(y.grad_fn)
print("++++++++++++++++++++++++++++++++++++++++")

z = y * y * 3
out = z.mean()
print(z)
print(out)
print("++++++++++++++++++++++++++++++++++++++++")
'''
关于方法.requires_grad_(): 该方法可以原地改变Tensor的属性.requires_grad的值. 如果没有主动设定默认为False.
'''
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
print("++++++++++++++++++++++++++++++++++++++++")
# 当代码要进行反向传播的时候, 直接调用.backward()就可以自动计算所有的梯度. 在这个Tensor上的所有梯度将被累加进属性.grad中.
out.backward()
print(x.grad)

# x的requires_grad = true,则x平方的requires_grad也是true

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
# 可以通过.detach()获得一个新的Tensor, 拥有相同的内容但不需要自动求导.
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
# 检测x y 是否拥有相同的内容(值比较矩阵)
print(x.eq(y).all())