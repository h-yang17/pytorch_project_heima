# 如果服务器上已经安装了GPU和DUDA
import torch.cuda
x = torch.rand(3, 5)
if torch.cuda.is_available():
    # 定义一个设备对象，这里指定为CUDA，即GPU
    device = torch.device("cuda")
    # 直接在GPU上创建一个tensor
    y = torch.ones_like(x, device=device)
    # 将cpu上的x张量移动到GPU上
    x = x.to(device)
    # x y 都在GPU上才支持加法运算
    z = x + y
    # z也在gpu上
    print(z)
    # 将z 转移到cpu上，并且指定张量元素的数据类型
    print(z.to("cpu", torch.double))
