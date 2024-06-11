"""
np.array 是 NumPy 库中最基本和最重要的功能之一，用于创建数组对象。NumPy 数组（ndarray）是一个多维的、均匀的数据结构，可以用于高效地存储和操作大型数据集。下面详细介绍 np.array 的用法和功能，并提供一些示例。
"""
import numpy as np

# 创建 NumPy 数组
# 1. 从列表或元组创建数组
# 从列表创建一维数组
a = np.array([1, 2, 3, 4, 5])
print(a)

# 从元祖创建一维数组
b = np.array((6, 7, 8, 9, 10))
print(b)

# 从嵌套列表创建二维数组
c = np.array([[1, 2, 3], [4, 5, 6]])
print(c)

# 2. 使用其他类型的数据创建数组
# 从一个范围创建数组
d = np.array(range(10))
print(d)

# 从另一个 NumPy 数组创建数组（复制）
e = np.array(d)
print("复制数组为\n",e)

# 数组的属性
# 创建数组后，可以使用一些属性来获取数组的信息：
print(a.ndim)   # 维度
print(a.shape)  # 形状
print(a.size)   # 元素总数
print(a.dtype)  # 数据类型


# 数组操作
# 1. 数组切片和索引
# 一维数组索引
print(a[0]) # 第一个元素
print(a[-1])# 最后一个元素

# 一维数组切片
print(a[1:3])# 第二到第三个元素

# 二维数组索引
print(c[0,0]) # 第一行第一列的元素
print(c[1,2])# 第二行第三列的元素

# 二维数组切片
print(c[:,1]) # 所有行的第二列
print(c[0,:]) # 第一行的所有列

# 2. 数组形状变换
f = np.array([1, 2, 3, 4, 5, 6])
g = f.reshape((2, -1)) # 将一维数组转换为二维数组 ; -1表示自己觉得另一维度是多少
print(g)

# 转置数组
h =c.T  # 将二维数组进行转置
print(h)

# 3. 数组的基本运算

# 数组元素的加减乘除运算
i = np.array([1, 2, 3])
j = np.array([4, 5, 6])
print(i + j)  # 元素加
print(i - j)  # 元素减
print(i * j)  # 元素乘
print(i / j)  # 元素除

# 数组与标量的运算
print(i * 2)  # 每个元素乘以 2

# 数组的矩阵乘法
k = np.array([[1, 2], [3, 4]])
l = np.array([[5, 6], [7, 8]])
print(np.dot(k, l)) # 矩阵乘法

"""
总结
np.array 是创建和操作 NumPy 数组的基础。NumPy 数组是一个高效的数据结构，支持多维度和各种类型的数据，并且提供了丰富的操作和函数，可以用于各种科学计算和数据分析任务。
"""