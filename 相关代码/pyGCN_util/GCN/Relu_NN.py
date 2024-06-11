"""
ReLU（Rectified Linear Unit，修正线性单元）是神经网络中广泛使用的一种激活函数。其数学表达式为：

f(x)=max(0,x)
ReLU激活函数的特点和优点
简单高效：ReLU函数简单且计算效率高。它只是对输入进行截断（将负值设为零），因此计算非常快。

稀疏激活：ReLU会导致一些神经元输出为零，从而引入稀疏性，这种稀疏性在某些情况下可以提高模型的泛化能力。

梯度消失问题的缓解：相比于Sigmoid和Tanh激活函数，ReLU有效地缓解了梯度消失问题，使得梯度在反向传播过程中更容易保持较大的值，从而加速模型的收敛。

ReLU的缺点
死亡ReLU问题：在训练过程中，如果学习率过大，可能会导致某些神经元的权重更新使得它们的输出永远为零，这些神经元就“死亡”了，无法再被激活。

输出不均匀：ReLU函数会将所有负值输入转换为零，这可能导致模型在处理某些数据时，输出不均匀，影响模型性能。

变体ReLU
为了克服ReLU的一些缺点，研究人员提出了多种ReLU的变体，如：

1、Leaky ReLU：引入了一个小的斜率，用于处理负值输入，公式为：

f(x)=  {
            x   if x≥0
            αx  if x<0
        }

其中，α是一个很小的常数（例如0.01）。

2、Parametric ReLU (PReLU)：类似于Leaky ReLU，但α是一个可以学习的参数。

Exponential Linear Unit (ELU)：通过指数函数处理负值部分，公式为：


f(x)={
        x   if x≥0
        α(exp(x)−1) if x<0
    }
ELU在负值部分有非零渐近值，从而改善了训练的稳定性。

这些变体在不同的情况下可以更好地处理ReLU的一些问题，从而提升模型的表现。
"""
# 示例
# 1. ReLU
import torch
import torch.nn as nn

# 定义一个包ReLU激活函数的简单神经网络
class SimpleNN_ReLU(nn.Module):
    def __init__(self):
        super(SimpleNN_ReLU, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relux
        x = self.layer2(x)
        return x

# 创建模型实例
model = SimpleNN_ReLU()
print(model)

# 2. Leaky ReLU
# 定义一个包含Leaky ReLU激活函数的简单神经网络
class SimpleNN_Leaky_ReLU(nn.Module):
    def __init__(self):
        super(SimpleNN_Leaky_ReLU, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.01)# 设置Leaky ReLU的负斜率
        self.layer2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.leaky_relu(x)
        x = self.layer2(x)
        return x

# 创建模型实例
model = SimpleNN_ReLU()
print(model)

# 3. Parametric ReLU (PReLU)

# 定义一个包含PReLU激活函数的简单神经网络
class SimpleNN_PReLU(nn.Module):
    def __init__(self):
        super(SimpleNN_PReLU,self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.prelu = nn.PReLU()# PReLU的参数是可学习的
        self.layer2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.prelu(x)
        x = self.layer2(x)
        return x

model = SimpleNN_PReLU()
print(model)


# 4. Exponential Linear Unit (ELU)

# 定义一个包含ELU激活函数的简单神经网络
class SimpleNN_ELU(nn.Module):
    def __init__(self):
        super(SimpleNN_ELU, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.elu = nn.ELU(alpha=1.0) # 设置ELU的alpha值
        self.layer2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.elu(x)
        x = self.layer2(x)
        return x

model = SimpleNN_ELU()
print(model)











