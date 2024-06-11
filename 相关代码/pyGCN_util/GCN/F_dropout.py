"""
F.dropout 是 PyTorch 中的一个函数，用于在训练神经网络时对输入应用 dropout 操作。Dropout 是一种正则化技术，通过随机将一部分神经元的输出设置为 0，从而防止模型过拟合。Dropout 操作只在训练期间生效，在测试期间不应用。

F.dropout 函数的定义
F.dropout 函数位于 torch.nn.functional 模块中，通常用作前向传播的一部分。其主要参数如下：

input: 要应用 dropout 的输入张量。
p: 随机将元素置零的概率，值在 [0, 1) 之间。默认值为 0.5。
training: 一个布尔值，表示是否处于训练模式。如果为 True，则应用 dropout；否则不应用。
inplace: 一个布尔值，如果为 True，则在输入张量上执行操作，而不是返回一个新的张量。
"""
# 示例代码
# 以下是如何在神经网络中使用 F.dropout 的示例。

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)  # 使用 F.dropout 进行 dropout 操作
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 示例模型实例
model = SimpleNN(nfeat=784, nhid=128, nclass=10, dropout=0.5)

# 打印模型结构
print(model)

# 创建一个示例输入
input_tensor = torch.randn(1, 784)

# 前向传播
output = model(input_tensor)
print(output)








