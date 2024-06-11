"""
torch.spmm 是 PyTorch 中用于稀疏矩阵和密集矩阵乘法的函数。它专门处理稀疏矩阵（通常是稀疏的 COO 或 CSR 格式）和密集矩阵之间的乘法操作。下面我们详细介绍其用法和示例。
用法
torch.spmm(sparse_matrix, dense_matrix)
sparse_matrix：稀疏矩阵，通常是 COO 或 CSR 格式的稀疏张量。
dense_matrix：密集矩阵，通常是一个二维的 dense 张量。
"""
import torch
import scipy.sparse
# 示例

# 创建一个稀疏矩阵（COO格式）
indices = torch.tensor([[0, 1, 1],
                        [2, 0, 2]])

values = torch.tensor([3, 4, 5], dtype=torch.float32)
size = torch.Size([2, 3])
sparse_matrix = torch.sparse_coo_tensor(indices, values, size)

# 创建一个密集矩阵
dense_matrix = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)

# 使用torch.spmm 进行稀疏矩阵和密集矩阵的乘法
result = torch.spmm(sparse_matrix, dense_matrix)
print(result)

"""
适用范围
稀疏矩阵格式：通常是 COO 或 CSR 格式。
密集矩阵：必须是二维的 dense 张量。
性能优势：torch.spmm 可以有效地处理大规模稀疏矩阵乘法，节省内存和计算时间。
注意事项
稀疏矩阵必须是 COO 或 CSR 格式。
密集矩阵必须是二维的 dense 张量。
矩阵的形状必须是兼容的，确保可以进行矩阵乘法。
"""

"""
处理 CSR 格式稀疏矩阵
如果你有一个 CSR 格式的稀疏矩阵，也可以使用 torch.spmm 进行计算。首先，需要将 CSR 格式转换为 COO 格式，因为 PyTorch 主要支持 COO 格式进行稀疏矩阵操作。
"""
# 示例
# 创建一个 CSR 格式的稀疏矩阵
csr_matrix = scipy.sparse.csr_matrix([[0, 0, 3], [4, 0, 5]])

# 转换为COO格式
coo_matrix = csr_matrix.tocoo()
values = torch.tensor(coo_matrix.data, dtype=torch.float32)
indices = torch.tensor([coo_matrix.row, coo_matrix.col], dtype=torch.int64)
size = torch.Size(coo_matrix.shape)
sparse_matrix = torch.sparse_coo_tensor(indices, values, size)

# 创建一个密集矩阵
dense_matrix = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)

# 使用torch.spmm进行稀疏矩阵和密集矩阵的乘法
result = torch.spmm(sparse_matrix, dense_matrix)
print(result)
# 通过上述示例，可以清楚地看到如何使用 torch.spmm 进行稀疏矩阵和密集矩阵的乘法操作。在处理大规模稀疏矩阵时，使用 torch.spmm 可以显著提高计算效率和节省内存。

