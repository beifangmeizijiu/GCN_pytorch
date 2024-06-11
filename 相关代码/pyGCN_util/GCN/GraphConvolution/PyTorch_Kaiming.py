"""
使用 PyTorch 实现
在 PyTorch 中，使用内置的初始化方法 kaiming_normal_ 和 kaiming_uniform_ 更加方便。这些方法实现了上述数学公式，并且可以直接应用于张量。
"""
import torch
import torch.nn as nn
import math
# Kaiming均匀分布初始化
tensor = torch.empty(2, 5)
nn.init.kaiming_uniform_(tensor, a = math.sqrt(6))
print(tensor)

# Kaiming 正态分布初始化
tensor = torch.empty(2, 5)
nn.init.kaiming_normal_(tensor)
print(tensor)










