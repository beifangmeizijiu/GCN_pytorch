"""
multiply 是一种矩阵操作方法，用于执行元素间的逐个相乘操作。在许多编程语言和库中都有类似的操作。下面以 numpy 和 scipy.sparse 为例，详细说明 multiply 的用法及其在处理矩阵时的作用。

numpy.multiply
numpy.multiply 是 numpy 库中的一个函数，用于逐元素相乘。
"""

import numpy as np
from scipy.sparse import coo_matrix
# 定义两个 numpy 数组
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8, 9], [10, 11, 12]])

# 逐元素相乘
C = np.multiply(A, B)

print("A:\n", A)
print("B:\n", B)
print("A * B:\n", C)

"""
scipy.sparse 中的 multiply
在 scipy.sparse 中，multiply 用于稀疏矩阵的逐元素相乘操作。
"""

# 定义两个稀疏矩阵
A = coo_matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
B = coo_matrix([[0, 4, 0], [0, 0, 5], [6, 0, 0]])

# 逐元素相乘
C = A.multiply(B)

print("A:\n", A.toarray())
print("B:\n", B.toarray())
print("A * B:\n", C.toarray())

"""
总结：
multiply 操作是逐元素相乘，适用于矩阵和稀疏矩阵的元素级别操作。
在 numpy 中，np.multiply 用于逐元素相乘。
在 scipy.sparse 中，multiply 用于稀疏矩阵的逐元素相乘。
对于邻接矩阵的对称化操作，multiply 可以用来逐元素比较和处理非零元素的位置关系。
"""







