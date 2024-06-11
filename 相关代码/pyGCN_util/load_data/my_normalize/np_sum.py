"""
sum 是一个函数，用于计算矩阵或多维数组的元素总和。根据不同的库，sum 函数的具体实现和用法可能略有不同。以下是一些常见的使用场景和示例。

"""
"""
NumPy 中的 sum 函数
在 NumPy 中，np.sum 用于计算数组元素的总和。可以指定沿哪个轴进行求和，或者计算整个数组的总和。
"""
import numpy as np
import torch

# 示例 1: 基本用法

# 创建一个数组
a = np.array([1, 2, 3, 4, 5])

# 计算数组的总和
total_sum = np.sum(a)

print(total_sum)

# 示例 2: 沿指定轴进行求和

# 创建一个二维数组
a = np.array([[1, 2, 3], [4, 5, 6]])

# 计算所有元素的总和
total_sum = np.sum(a)
print("Total sum:", total_sum)

# 沿行方向（轴0）求和（计算每一列的和）
sum_along_axis_0 = np.sum(a,axis=0)
print("Sum along axis 0:",sum_along_axis_0)

# 沿列方向（轴1）求和（计算每一行的和）
sum_along_axis_1 =np.sum(a, axis=1)
print("Sum along axis 1:",sum_along_axis_1)

# 示例 3: 带初始值的求和

# 创建一个数组
a = np.array([1, 2, 3, 4, 5])

# 计算数组的总和，并添加一个初始值10
total_sum_with_inital = np.sum(a,initial=10)
print(total_sum_with_inital)

# 示例 4: 处理 NaN 值
# 创建一个包含NaN的数组
a = np.array([1, 2, np.nan, 4, 5])

# 计算数组的总和，忽略NaN值
total_sum = np.nansum(a)

print('nan=',total_sum)

# 示例 5: 对布尔数组求和

# 创建一个布尔数组
a = np.array([True, False, True, True])

# 计算数组中True的个数
count_true = np.sum(a)

print(count_true)
"""
总结
np.sum 是一个非常灵活且强大的函数，用于计算数组元素的总和。无论是对整个数组求和，还是沿指定轴求和，甚至处理包含 NaN 的数组，np.sum 都能提供
"""


# PyTorch 中的 sum 函数
# 在 PyTorch 中，torch.sum 用于计算张量元素的总和。同样可以指定沿哪个维度进行求和。

# 创建一个二维张量
a = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 计算所有元素的总和
total_sum = torch.sum(a)
print("Total sum:", total_sum)

# 沿行方向（维度0）求和
sum_along_dim_0 = torch.sum(a, dim=0)
print("Sum along dim 0:", sum_along_dim_0)

# 沿列方向（维度1）求和
sum_along_dim_1 = torch.sum(a, dim=1)
print("Sum along dim 1:", sum_along_dim_1)






