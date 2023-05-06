'''
关于torch.nn:
使用Pytorch来构建神经网络, 主要的工具都在torch.nn包中.
nn依赖于autograd来定义模型, 并对其自动求导.

构建神经网络的典型流程:
定义一个拥有可学习参数的神经网络
遍历训练数据集
处理输入数据使其流经神经网络
计算损失值
将网络参数的梯度进行反向传播
以一定的规则更新网络的权重

激活函数Relu，在神经网络中的作用是：通过加权的输入进行非线性组合产生非线性决策边界 简单的来说就是增加非线性作用。
在深层卷积神经网络中使用激活函数同样也是增加非线性，主要是为了解决sigmoid函数带来的梯度消失问题。

在PyTorch中对于不能整除的状况默认均为向下取整，可以选择向上取整
'''
import torch
# 导入若干工具包
import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义一个简单的网络类
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 定义第一层卷积神经网络, 输入通道维度=1, 输出通道维度=6, 卷积核大小3*3
        self.conv1 = nn.Conv2d(1, 6, 3)
        # 定义第二层卷积神经网络, 输入通道维度=6, 输出通道维度=16, 卷积核大小3*3
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 定义三层全连接网络
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 在(2, 2)的池化窗口下执行最大池化操作
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # 计算size, 除了第0个维度上的batch_size
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())

'''
orch.randn[8, 3, 244, 244]，[batch, channel, height, width]，表示batch_size=8， 3通道（灰度图像为1），图片尺寸：224x224
如果给的是torch.randn[1, 1, 32, 32]表示batch_size=1， 1通道（灰度图像），图片尺寸：32x32
'''
input = torch.randn(1, 1, 32, 32)
print("+++++++++++++++++")
print(input)
print(input.size())
print("+++++++++++++++++")
out = net(input)
print(out)
print(out.size())
'''
torch.nn构建的神经网络只支持mini-batches的输入, 不支持单一样本的输入.
比如: nn.Conv2d 需要一个4D Tensor, 形状为(nSamples, nChannels, Height, Width). 如果你的输入只有单一样本形式,
 则需要执行input.unsqueeze(0), 主动将3D Tensor扩充成4D Tensor.
'''
'''
损失函数的输入是一个输入的pair: (output, target), 然后计算出一个数值来评估output和target之间的差距大小.
在torch.nn中有若干不同的损失函数可供使用, 比如nn.MSELoss就是通过计算均方差损失来评估输入和目标值之间的差距.
'''
output = net(input)
target = torch.randn(10)
print("+++++++++++++++++")
print(target)
print(target.size())
print("+++++++++++++++++")
# 改变target的形状为二维张量, 为了和output匹配
target = target.view(1, -1)
print("+++++++++++++++++")
print(target)
print(target.size())
print("+++++++++++++++++")
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
'''
关于方向传播的链条: 如果我们跟踪loss反向传播的方向, 使用.grad_fn属性打印, 将可以看到一张完整的计算图如下:
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
'''
'''
当调用loss.backward()时, 整张计算图将对loss进行自动求导, 
所有属性requires_grad=True的Tensors都将参与梯度求导的运算, 并将梯度累加到Tensors中的.grad属性中.
'''
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

'''
反向传播(backpropagation)
在Pytorch中执行反向传播非常简便, 全部的操作就是loss.backward().
在执行反向传播之前, 要先将梯度清零, 否则梯度会在不同的批次数据之间被累加.
'''
# Pytorch中执行梯度清零的代码
net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

# Pytorch中执行反向传播的代码
loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


# 首先导入优化器的包, optim中包含若干常用的优化算法, 比如SGD, Adam等
import torch.optim as optim

# 通过optim创建优化器对象
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 将优化器执行梯度清零的操作
optimizer.zero_grad()

output = net(input)
loss = criterion(output, target)

# 对损失值执行反向传播的操作
loss.backward()
# 参数的更新通过一行标准代码来执行
optimizer.step()

print("hello world")