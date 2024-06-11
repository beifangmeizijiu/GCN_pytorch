"""
1. Kaiming 正态分布初始化
我们使用标准正态分布生成随机数，并根据公式缩放方差。
公式：w∼N(0, sqrt(2/n))其中 n 是输入层的单元数。
"""
import numpy as np

def kaming_normal(shape, fan_in):
    std = np.sqrt(2.0 / fan_in) ** 0.5
    return np.random.normal(0, std, shape)

# 示例：初始化一个形状为 (out_features, in_features) 的权重矩阵
in_features = 5
out_features = 2
weight = kaming_normal((out_features, in_features), fan_in = in_features)
print(weight)

"""
2. Kaiming 均匀分布初始化
我们使用均匀分布生成随机数，并根据公式设置范围。
Kaiming 均匀分布初始化：
w∼U(-sqrt(6), sqrt(6))
其中 w 是权重，n 是输入层的单元数。
"""
def kaiming_uniform(shape, fan_in, a=0):
    bound = np.sqrt(6.0 / fan_in)
    return np.random.uniform(-bound, bound, shape)

# 示例：初始化一个形状为 (out_features, in_features) 的权重矩阵
in_features = 5
out_features = 2
weight = kaiming_uniform((out_features, in_features), fan_in=in_features)
print(weight)

