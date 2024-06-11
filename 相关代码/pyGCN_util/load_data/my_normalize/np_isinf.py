"""
np.isinf 是 NumPy 提供的一个函数，用于检测数组中的元素是否为正无穷或负无穷。它返回一个布尔数组，数组中每个位置的值为 True 表示该位置的元素为无穷大，False 表示该位置的元素不是无穷大。
"""
import numpy as np

# 示例 1: 基本用法
# 创建一个包含无穷大的数组
a = np.array([1, 2, np.inf, -np.inf, 5])

# 检测无穷大
inf_mask = np.isinf(a)

print(inf_mask)
# 在这个例子中，数组 a 中第三个元素为正无穷大，第四个元素为负无穷大。np.isinf(a) 返回一个布尔数组，指示哪些位置的元素为无穷大。

# 示例 2: 使用 np.isinf 处理无穷大值

# 替换无穷大值为0
a[np.isinf(a)] = 0
print(a)
# 在这个例子中，我们使用 np.isinf(a) 创建一个布尔掩码，然后使用该掩码将数组中所有的无穷大值替换为0。

# 示例 3: 检测二维数组中的无穷大值

# 创建一个包含无穷大的二维数组
a = np.array([[1, 2, np.inf], [np.inf, 5, -np.inf]])

# 检测无穷大
inf_mask = np.isinf(a)

print(inf_mask)

# 在这个例子中，np.isinf 同样适用于二维数组，返回一个同样形状的布尔数组。

# 示例 4: 与其他函数结合使用

# 计算无穷大值的数量
inf_count = np.sum(np.isinf(a))

print(inf_count)

# 在这个例子中，np.sum(np.isinf(a)) 计算数组 a 中无穷大值的数量。

# 总结
# np.isinf 是一个非常有用的函数，可以帮助你检测数组中的无穷大值，并对这些值进行进一步处理。无论是过滤、替换还是统计，np.isinf 都能提供有效的帮助。





