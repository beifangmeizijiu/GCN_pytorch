"""
scipy.sparse.csr_matrix 是 SciPy 库中用于创建稀疏矩阵的一种格式。CSR 代表压缩稀疏行（Compressed Sparse Row），它是一种高效的存储稀疏矩阵的方式，特别适合进行快速的行切片操作和矩阵向量乘法。
"""
import numpy as np
import scipy.sparse as sp

# 从密集矩阵创建
dense_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 0, 0]])
sparse_matrix = sp.csr_matrix(dense_matrix)
print(type(sparse_matrix),'\n',sparse_matrix)

# 从坐标格式 (COO) 创建
data = np.array([1, 2, 3, 4])
row_indices = np.array([0, 0, 1, 2])
col_indices = np.array([0, 2, 2, 0])

sparse_matrix = sp.csr_matrix((data, (row_indices, col_indices)), shape=(3, 3))
print(f'坐标格式 (COO) 创建\n{sparse_matrix}')

# 从其他系数格式转换
# COO 格式
data = np.array([1, 2, 3, 4])
row_indices = np.array([0, 0, 1, 2])
col_indices = np.array([0, 2, 2, 0])

coo = sp.coo_matrix((data, (row_indices, col_indices)), shape=(3, 3))
print('coo格式',coo)
# 转换为CSR格式
sparse_matrix = coo.tocsr()
print(f'从其他系数格式转换\n{sparse_matrix}')

"""
CSR 矩阵的属性和方法
属性
data: 非零元素的数组。
indices: 列索引的数组。
indptr: 行指针的数组，用于指示每一行的起始位置。
shape: 矩阵的形状。
nnz: 非零元素的个数。
方法
toarray(): 将稀疏矩阵转换为密集（NumPy）数组。
tocsr(): 将矩阵转换为 CSR 格式。
tocsc(): 将矩阵转换为 CSC 格式（压缩稀疏列）。
transpose(): 转置矩阵。
multiply(): 元素级乘法。
dot(): 矩阵乘法。
"""
# 矩阵向量乘法
dense_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 0, 0]])
sparse_matrix = sp.csr_matrix(dense_matrix)

vector = np.array([1, 2, 3])
result = sparse_matrix.dot(vector)
print("矩阵乘法结果：",result)

# 转换为密集数组
dense_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 0, 0]])
sparse_matrix = sp.csr_matrix(dense_matrix)

dense_array = sparse_matrix.toarray()
print(dense_array)

