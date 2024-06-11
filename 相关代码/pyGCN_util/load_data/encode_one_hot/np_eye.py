"""
np.eye 是 NumPy 中的一个函数，用于创建一个对角线元素为 1，其余元素为 0 的二维数组（即单位矩阵）。它的常见用法包括生成方阵、非方阵和带有偏移对角线的矩阵。

函数定义
numpy.eye(N, M=None, k=0, dtype=<class 'float'>, order='C')
参数说明
N: 输出矩阵的行数。
M: 输出矩阵的列数。如果未提供，默认为 N（生成方阵）。
k: 对角线的索引。k=0 表示主对角线，k>0 表示上对角线，k<0 表示下对角线。
dtype: 输出矩阵的数据类型。默认值为 float。
order: 存储多维数据的顺序。'C' 表示行优先（C 风格），'F' 表示列优先（Fortran 风格）。

"""
import numpy as np
# 示例 生成 3x3 单位矩阵
I = np.eye(3)
print(f"单位矩阵\n{I}")
# 示例 生成 4x3 矩阵
I = np.eye(4, 3)
print(f"4*3单位矩阵\n{I}")
# 示例 生成 3x3 矩阵，主对角线偏移 1
I = np.eye(3, k=1)
print(f"单位矩阵向上偏移1\n{I}")
# 示例 生成 3x3 矩阵，主对角线偏移 2
I = np.eye(3, k=2)
print(f"单位矩阵向上偏移2\n{I}")
# 示例 生成 3x3 矩阵，主对角线偏移 -1
I = np.eye(3, k=-1)
print(f"单位矩阵向上偏移-1\n{I}")
# 示例 生成 3x3 矩阵，主对角线偏移 -8（相当于给1移除了）
I = np.eye(3, k=-8)
print(f"单位矩阵向上偏移-8\n{I}")

"""
在矩阵中，对角线的偏移（上对角线和下对角线）是相对于主对角线的位置而言的。主对角线是指从矩阵的左上角到右下角的那条对角线。

主对角线：默认情况下，即 𝑘 = 0。
上对角线：对角线偏移为正数，即 k>0。
这意味着对角线在主对角线上方，向右移动 k 列。
下对角线：对角线偏移为负数，即 k<0。
这意味着对角线在主对角线下方，向左移动 k 列。
"""
# 示例 假设我们有一个 4×4 的矩阵，说明对角线的偏移：
# 主对角线 k = 0;
# 主对角线 k=0：从左上到右下的对角线。
I = np.eye(4, k=0)
print("主对角线 k=0:")
print(I)

# 上对角线 k = 1
# 上对角线 k=1：对角线在主对角线上方，向右移动一列。

I = np.eye(4, k=1)
print("\n上对角线 k=1:")
print(I)

# 下对角线 k = -1
# 下对角线 k=−1：对角线在主对角线下方，向左移动一列。

I = np.eye(4, k=-1)
print("\n下对角线 k=-1:")
print(I)
"""
C风格与F风格
在 NumPy 中，'C' 表示行优先（C 风格），'F' 表示列优先（Fortran 风格）是指数组在内存中存储数据的顺序。

行优先（C 风格）
行优先存储方式意味着在内存中，数组的行是连续存储的。
在这种情况下，对于二维数组，先按行存储数据，然后再存储下一行。
列优先（Fortran 风格）
列优先存储方式意味着在内存中，数组的列是连续存储的。
在这种情况下，对于二维数组，先按列存储数据，然后再存储下一列。
"""
# 示例 下面是一个示例，展示如何使用这两种存储方式以及它们的区别：
# 创建一个 2x3 的数组
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 默认是 C 风格（行优先）
arr_c = np.array(arr,order='C')

# Fortran 风格（列优先）
arr_f = np.array(arr, order='F')

print("原始数组：")
print(arr)

print("\nC 风格（行优先）存储：")
print(arr_c)

print("\nFortran 风格（列优先）存储：")
print(arr_f)

# 检查内存中的数据顺序
print("\nC 风格（行优先）存储的数据顺序：")
print(arr_c.flatten(order='K'))

print("\nFortran 风格（列优先）存储的数据顺序：")
print(arr_f.flatten(order='K'))

"""
这种内存存储方式的选择会影响数组的访问速度和存储效率，尤其在处理大规模数组或进行复杂的数值计算时需要加以考虑。
"""












