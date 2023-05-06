import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 使用torchvision下载CIFAR10数据集
'''
下载数据集并对图片进行调整,
因为torchvision数据集的输出是PILImage格式, 数据域在[0, 1]. 我们将其转换为标准数据域[-1, 1]的张量格式.
'''
# 利用transforms.compose进行转换，转换为tensor类型
# 下面是标准代码，定义数据转换器
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
# num_workers=2是两个线程，多个线程加速读取数据速度
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
# 测试时不需要打乱，所以shuffle=False
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 仿照7.1节中的类来构造此处的类, 唯一的区别是此处采用3通道3-channel

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)

'''
Momentum的作用？
主要是在训练网络时，最开始会对网络进行权值初始化，但是这个初始化不可能是最合适的；
因此可能就会出现损失函数在训练的过程中出现局部最小值的情况，而没有达到全局最优的状态。
momentum的出现可以在一定程度上解决这个问题。动量来源于物理学，当momentum越大时，
转换为势能的能量就越大，就越有可能摆脱局部凹区域，从而进入全局凹区域。momentum主要是用于权值优化。
'''

# 采用交叉熵损失函数和随机梯度下降优化器.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # data中包含输入图像张量inputs, 标签张量labels
        inputs, labels = data

        # 首先将优化器梯度归零
        optimizer.zero_grad()

        # 输入图像张量进网络, 得到输出张量outputs
        outputs = net(inputs)

        # 利用网络的输出outputs和标签labels计算损失值
        loss = criterion(outputs, labels)

        # 反向传播+参数更新, 是标准代码的标准流程
        loss.backward()
        optimizer.step()

        # 打印轮次和损失值
        running_loss += loss.item()
        if (i + 1) % 2000 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# 首先设定模型的保存路径
PATH = './cifar_net.pth'
# 保存模型的状态字典
torch.save(net.state_dict(), PATH)

# 先拿出四张图片进行简单的测试
# 先取出4张图片
dataiter = iter(testloader)
# 两种利用迭代器的方法
# images, labels = next(dataiter)
images, labels = dataiter.__next__()
# 首先实例化模型的类对象
net = Net()
# 加载训练阶段保存好的模型的状态字典
net.load_state_dict(torch.load(PATH))
# 利用模型对图片进行预测
outputs = net(images)
# 共有10个类别, 采用模型计算出的概率最大的作为预测的类别
# dim=0表示计算每列的最大值，dim=1表示每行的最大值
_, predicted = torch.max(outputs, 1)
# 打印预测标签的结果
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# 在全部测试集上运行
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        # data中有四个数据
        images, labels = data
        '''
        torch.Size括号中有几个数字就是几维
        例如：第一层（最外层）中括号里面包含了两个中括号（以逗号进行分割），这就是（2，3，4）中的2
        第二层中括号里面包含了三个中括号（以逗号进行分割），这就是（2，3，4）中的3
        第三层中括号里面包含了四个数（以逗号进行分割），这就是（2，3，4）中的4
        '''
        print("++++++++++++++")
        print(labels)
        print(type(labels))
        print(labels.size)
        print(labels.size())
        # 范围只能是-1 到0
        # 具体原因看main.py
        print(labels.size(-1))
        print(labels.size(0))
        print("++++++++++++++")
        outputs = net(images)
        # dim=0表示计算每列的最大值，dim=1表示每行的最大值
        # 返回值有两个，取后者。取返回最大值所在的索引
        _, predicted = torch.max(outputs.data, 1)
        # labels.size(0)
        total += labels.size(0)
        # predicted和labels都是一行四列的值。sum把其中的值为TRUE的加起来
        correct += (predicted == labels).sum().item()
# %% 字符%
print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
# 为了更加细致的看一下模型在哪些类别上表现更好, 在哪些类别上表现更差, 我们分类别的进行准确率计算.
class_correct = list(0. for i in range(10))
# class_correct中有10个0.0
class_total = list(0. for i in range(10))
# class_total中有10个0.0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        print(predicted)
        print(predicted.size())
        print(predicted == labels)
        print((predicted == labels).size())
        # 把.squeeze()删除也能运行
        # https://zhuanlan.zhihu.com/p/368920094
        c = (predicted == labels).squeeze()
        print(c)
        print(c.size())
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
