"""
scipy.sparse 模块中的稀疏矩阵对象有许多属性和方法，用于处理和操作稀疏矩阵。以下是一些常见的属性及其功能：

常见稀疏矩阵对象
csr_matrix：压缩行存储（Compressed Sparse Row）格式。
csc_matrix：压缩列存储（Compressed Sparse Column）格式。
coo_matrix：坐标列表（Coordinate List）格式。
dia_matrix：对角线（Diagonal）格式。
dok_matrix：字典（Dictionary of Keys）格式。
lil_matrix：列表（List of Lists）格式。
共同属性
这些稀疏矩阵对象共享一些常见属性：
"""
import numpy as np
import scipy.sparse as sp

# 创建一个 CSR 格式的稀疏矩阵
csr_matrix = sp.csr_matrix([[0, 1, 0], [2, 0, 3]])

# 打印 CSR 矩阵的属性
print("CSR Matrix:")
# 1、shape：
#
# 表示矩阵的维度（行数和列数）。
# 示例:
print("Shape:", csr_matrix.shape)
# 2、ndim：
#
# 表示矩阵的维度数（对于稀疏矩阵，总是 2）。
# 示例：
print("Number of dimensions:", csr_matrix.ndim)
# 3、nnz：
#
# 表示矩阵中的非零元素数量。
# 示例：
print("Number of non-zero elements:", csr_matrix.nnz)
# 4、data：
#
# 存储矩阵中的非零元素。
# 示例：
print("Data array:", csr_matrix.data)
# 5、indices（适用于 CSR 和 CSC 格式）：
#
# 表示非零元素的列索引（CSR）或行索引（CSC）。
# 示例：
print("Indices array:", csr_matrix.indices)
# 6、indptr（适用于 CSR 和 CSC 格式）：
#
# 表示行（CSR）或列（CSC）的指针，指向 indices 和 data 数组的起始位置。
# 示例：
print("Index pointer array:", csr_matrix.indptr)
# 7、dtype：
#
# 表示矩阵元素的数据类型。
# 示例：
print("Data type:", csr_matrix.dtype)

# 创建一个 COO 格式的稀疏矩阵
coo_matrix = sp.coo_matrix([[0, 1], [2, 0]])

# 打印 COO 矩阵的属性
print("\nCOO Matrix:")
print("Shape:", coo_matrix.shape)
print("Number of dimensions:", coo_matrix.ndim)
print("Number of non-zero elements:", coo_matrix.nnz)
print("Data array:", coo_matrix.data)
# 6、row 和 col（适用于 COO 格式）：
#
# 分别表示非零元素的行和列索引。
# 示例：
print("Row indices array:", coo_matrix.row)
print("Column indices array:", coo_matrix.col)
print("Data type:", coo_matrix.dtype)



