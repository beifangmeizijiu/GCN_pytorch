"""
torch.sparse 是 PyTorch 提供的处理稀疏张量的模块。稀疏张量在大多数元素为零的情况下非常有用，因为它们可以显著减少内存使用和计算时间。PyTorch 支持稀疏 COO（Coordinate List）格式和 CSR（Compressed Sparse Row）格式的张量。
"""
import torch
"""
COO 格式
COO（Coordinate List）格式是一种常见的稀疏矩阵表示法，它通过三个一维数组来表示矩阵：行索引数组、列索引数组和对应位置的值数组。PyTorch 支持使用 COO 格式来创建稀疏张量。
"""

# 创建 COO 格式的稀疏张量

# 定义稀疏张量索引和值
indices = torch.tensor([[0, 1, 1],
                     [2, 0, 2]]) # 2x3的索引矩阵（行索引，列索引）

values = torch.tensor([3, 4, 5], dtype=torch.float32) # 非零元素的值

# 创建稀疏张量
sparse_tensor = torch.sparse_coo_tensor(indices, values, (2, 3))

# 打印稀疏张量
print(sparse_tensor)

# COO 格式稀疏张量操作
# 你可以对稀疏张量执行一些常见的操作，例如矩阵乘法、加法等。
# 矩阵乘法
# 定义一个密集张量
dense_tensor = torch.tensor([[1, 2], [3, 4], [5, 6]],dtype= torch.float32)

# 执行矩阵乘法
result = torch.sparse.mm(sparse_tensor,dense_tensor)

print(result)

# 转换为密集张量
# 你可以将稀疏张量转换为密集张量，以便进行进一步操作或检查其内容。
dense_result = sparse_tensor.to_dense()
print(dense_result)

"""
稀疏张量与自动求导
稀疏张量支持 PyTorch 的自动求导机制，但需要注意的是，不是所有的操作都支持稀疏张量的自动求导。因此，在使用稀疏张量时，需要检查所使用的操作是否支持自动求导。

总结
torch.sparse 模块提供了处理稀疏张量的强大工具，包括 COO 和 CSR 格式的支持。利用这些工具，可以有效地处理大规模稀疏数据，显著提高内存和计算效率。在使用稀疏张量时，要熟悉其操作和限制，以充分发挥其优势。
"""


















