"""
np.vstack 是 NumPy 库中的一个函数，用于将一系列数组在垂直方向（按行顺序）堆叠在一起。vstack 可以处理一维或二维数组，将它们合并为一个更大的二维数组。
用法

语法
np.vstack(tup)

tup：一个元组或列表，包含要堆叠的数组。
返回值
返回一个新的数组，是输入数组沿垂直方向堆叠的结果。
"""
import numpy as np
# 示例
# 以下示例展示了如何使用 np.vstack 将多个数组堆叠在一起。
# 示例 1：堆叠一维数组
# 创建一维数组
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

# 使用np.vstack垂直堆叠
result = np.vstack((array1,array2))

print(result)

# 示例 2：堆叠二维数组

# 创建二维数组
array1 = np.array([[1, 2, 3], [4, 5, 6]])
array2 = np.array([[7, 8, 9], [10, 11, 12]])

# 使用 np.vstack 垂直堆叠
result = np.vstack((array1, array2))

print(result)

"""
注意事项
数组形状：被堆叠的数组必须具有相同的形状（除去堆叠轴之外的形状必须一致），即它们必须有相同的列数。

一维数组堆叠：当堆叠一维数组时，np.vstack 会先将一维数组转换为二维行向量，然后再进行堆叠。
"""












