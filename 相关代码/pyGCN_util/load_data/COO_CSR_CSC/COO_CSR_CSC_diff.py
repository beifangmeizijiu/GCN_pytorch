"""
COO（Coordinate）、CSR（Compressed Sparse Row）和CSC（Compressed Sparse Column）是三种常用的稀疏矩阵存储格式，它们在存储结构和适用场景上有一些区别。
"""
import numpy as np
import scipy.sparse as sp

# 定义矩阵的非零值
data = np.array([1, 2, 3, 4])
row = np.array([0, 0, 1, 2])
col = np.array([0, 2, 2, 0])


"""
COO（Coordinate Format）
COO格式是一种直观的稀疏矩阵存储格式，通过记录非零元素的位置及其值来表示稀疏矩阵。它使用三个数组：

data: 存储非零元素的值。
row: 存储非零元素所在的行索引。
col: 存储非零元素所在的列索引。

"""

# 创建COO稀疏矩阵
coo = sp.coo_matrix((data, (row, col)), shape=(3, 8))
print(coo)
print("矩阵原貌",coo.toarray())
"""
CSR（Compressed Sparse Row Format）
CSR格式通过压缩行索引来存储稀疏矩阵。它使用三个数组：

data: 存储非零元素的值。
indices: 存储非零元素的列索引。
indptr: 指示每一行的起始位置。
这种格式适用于高效的行切片和矩阵-向量乘法。
"""
# 从COO格式转换为CSR格式
csr = coo.tocsr()
print('csr\n',csr)

"""
CSC（Compressed Sparse Column Format）
CSC格式通过压缩列索引来存储稀疏矩阵。它也使用三个数组：

data: 存储非零元素的值。
indices: 存储非零元素的行索引。
indptr: 指示每一列的起始位置。
这种格式适用于高效的列切片和矩阵-向量乘法。
"""
csc = coo.tocsc()
print('csc\n',csc)

"""
COO格式:

优点: 直观且易于构建，适合于一次性构建和修改矩阵。
缺点: 访问和操作效率低。
应用场景: 构建和快速输入输出。
CSR格式:

优点: 行切片和矩阵-向量乘法高效。
缺点: 列切片不高效。
应用场景: 行切片、矩阵-向量乘法、高效存储。
CSC格式:

优点: 列切片和矩阵-向量乘法高效。
缺点: 行切片不高效。
应用场景: 列切片、矩阵-向量乘法、高效存储。
"""