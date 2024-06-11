"""
np.where 是 NumPy 中的一个函数，用于返回满足给定条件的元素的索引或根据条件选择元素。它的功能类似于条件筛选操作，可以用于许多应用场景。根据使用方式的不同，np.where 有以下几种主要用法：
"""
import numpy as np
# 1、返回满足条件元素的索引：当只传入一个条件时，np.where 返回满足条件的元素的索引。
# 示例数组
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 1])

# 找出大于5的元素的索引
indices = np.where(arr > 5)
print(indices)

# 2. 根据条件选择元素
# 当传入三个参数时，np.where(condition, x, y) 返回一个数组，其中满足条件的元素来自 x，不满足条件的元素来自 y。
# 示例数组
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 根据条件选择元素
result = np.where(arr > 5, arr, -1)
print(result)

# 3. 在二维数组中的应用
# 在处理二维数组时，np.where 也非常有用。例如，找出矩阵中满足特定条件的元素的索引：
# 示例二维数组
arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

# 找出大于4的元素的索引
indices = np.where(arr>4)
print(indices)

# 使用这些索引提取元素
print(arr[indices])

# 4. 结合使用 np.where 和 torch
# 在处理 PyTorch 张量时，np.where 也可以和 PyTorch 的张量操作结合使用。例如，从 PyTorch 张量中选择满足特定条件的元素：
import torch

# 示例Pytorch 张量
tensor = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

#将PyTorch张量转换为Numpy数组
arr = tensor.numpy()

# 使用np.where找出大于4的元素的索引
indices = np.where(arr>4)

# 使用这些索引在PyTorch张量中提取元素
result = tensor[indices]
print(result)
"""
总结
np.where 是一个非常强大的工具，用于根据条件筛选和操作数组。它可以返回满足条件的索引或根据条件选择元素，适用于一维和多维数组，也可以与 PyTorch 张量结合使用。在进行复杂的数据处理和分析时，np.where 能显著简化代码并提高可读性。
"""