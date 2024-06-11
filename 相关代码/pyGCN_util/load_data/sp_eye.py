"""
scipy.sparse.eye 是 SciPy 库中用于生成稀疏单位矩阵的函数。这个函数返回的是一个稀疏矩阵，而不是密集矩阵。稀疏矩阵在存储和计算上都更为高效，尤其是在处理大规模且大部分元素为零的矩阵时。
"""
import scipy.sparse as sp
# 以下是如何使用 scipy.sparse.eye 生成一个稀疏单位矩阵的示例：

# 生成一个5x5的稀疏单位矩阵
spare_identity_matrix = sp.eye(5)

print(spare_identity_matrix)

"""
这个输出表示一个5x5的稀疏矩阵，其中仅有对角线上的元素为1，其余元素为0。

其他格式的稀疏矩阵
默认情况下，scipy.sparse.eye 返回一个COO（Coordinate list）格式的稀疏矩阵，但你可以通过指定 format 参数来选择其他稀疏矩阵格式，例如CSR（Compressed Sparse Row）或CSC（Compressed Sparse Column）格式。

"""
# 例如，生成一个CSR格式的稀疏单位矩阵：
# 生成一个 5x5 的 CSR 格式稀疏单位矩阵
spare_identity_matrix_csr = sp.eye(5, format='csr')

print(spare_identity_matrix_csr)

"""
将稀疏矩阵转换为密集矩阵
如果你需要将稀疏矩阵转换为密集矩阵，可以使用 todense() 或 toarray() 方法。

"""
# 将稀疏矩阵转换为密集矩阵
dense_identity_matrix = spare_identity_matrix.todense()
print(dense_identity_matrix)

"""
总结
scipy.sparse.eye 是一个非常有用的函数，特别是在需要处理大型稀疏矩阵时。它生成的稀疏单位矩阵不仅在存储上更为高效，而且在计算上也能显著提高性能。如果需要将稀疏矩阵转换为密集矩阵，可以使用相应的方法进行转换。
"""












