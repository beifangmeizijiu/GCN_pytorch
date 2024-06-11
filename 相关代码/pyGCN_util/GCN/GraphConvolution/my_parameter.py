"""
Parameter 是 PyTorch 中用于定义模型参数的一个类。它通常用于将张量注册为模型的参数，以便在训练过程中进行优化。具体来说，torch.nn.Parameter 是一个张量的子类，它的主要特点是当 Parameter 对象被赋值给 nn.Module 的属性时，它会被自动添加到模型的参数列表中。
"""
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
# 用法
# 下面是一些常见的用法示例：
#
# 1. 定义自定义层
# 在定义自定义神经网络层时，你可以使用 Parameter 来声明可训练的参数。

class MyLinear(nn.Module):
    # 权重参数：self.weight 是一个大小为 (out_features, in_features) 的张量，通过 Parameter 类进行声明。
    # 偏置参数：如果 bias 为 True，则 self.bias 是一个大小为 out_features 的张量。如果 bias 为 False，则不使用偏置。
    def __init__(self, in_features, out_features, bias=True):
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features= out_features
        """
        在 PyTorch 中，Parameter 是 torch.Tensor 的一个子类。当你将一个 Parameter 对象赋值给 nn.Module 的属性时，它会被自动添加到模型的参数列表中，并在训练过程中进行优化。
        """
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.rest_parameters()

    # 重置参数
    def rest_parameters(self):
        # 使用 Kaiming 初始化方法来初始化权重参数
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(6))
        # 如果 self.bias 不为 None，则使用均匀分布来初始化偏置参数。
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1/ math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    # 前向传播
    # forward 方法定义了前向传播的计算过程：
    def forward(self, input):
        # 使用 torch.nn.functional.linear 函数进行线性变换，计算公式为：output = input * weight^T + bias。
        return torch.nn.functional.linear(input, self.weight, self.bias)

if __name__ == '__main__':
    # 使用自定义层
    input = torch.randn(3, 5)
    linear = MyLinear(5, 2)
    output = linear(input)
    print(output)


"""
nn.init._calculate_fan_in_and_fan_out 是 PyTorch 内部的一个工具函数，用于计算给定张量的 fan_in 和 fan_out，即输入和输出的单元数。这些数值在初始化权重时非常有用，尤其是对于如 Kaiming 初始化等需要根据输入或输出单元数来设置标准差或范围的方法。

使用示例
下面是如何使用 nn.init._calculate_fan_in_and_fan_out 函数来计算权重矩阵的 fan_in 和 fan_out。
"""
import torch
import torch.nn as nn

# 创建一个权重张量
weight = torch.empty(2, 5)

# fan_in 和 fan_out 的计算原理
# fan_in: 输入单元数，是权重张量形状的倒数第二个维度的大小。
# fan_out: 输出单元数，是权重张量形状的最后一个维度的大小。
# 计算 fan_in 和 fan_out
fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)

print("fan_in:", fan_in)
print("fan_out:", fan_out)

"""
nn.init.uniform_ 是 PyTorch 中用于均匀分布初始化的方法。它可以将张量的值初始化为在给定范围内均匀分布的随机数。下面我们详细说明它的用法及其在自定义层中的应用。

使用示例
以下示例展示了如何使用 nn.init.uniform_ 方法初始化张量
"""
import torch
import torch.nn as nn

# 创建一个未初始化的张量
tensor = torch.empty(3, 5)

# 将张量初始化为 [-0.5, 0.5] 范围内均匀分布的随机数
nn.init.uniform_(tensor, a=-0.5, b=0.5)
print(tensor)


