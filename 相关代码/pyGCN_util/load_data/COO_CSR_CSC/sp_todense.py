"""
在处理稀疏矩阵（sparse matrix）时，todense() 是一个常用的方法，用于将稀疏矩阵转换为密集矩阵（dense matrix）。在Python的科学计算库，如SciPy和NumPy中，这种转换方法是非常实用的，尤其是在处理图数据时，特征矩阵可能是以稀疏形式存储的。

假设你有一个稀疏矩阵 features，并且你想将它转换为密集矩阵，代码通常是这样的：
dense_features = features.todense()


"""

import numpy as np
from scipy.sparse import csr_matrix

# 创建一个稀疏矩阵

sparse_matrix = csr_matrix([[1, 0, 0], [0, 0, 2], [0, 3, 0]])

# 打印稀疏矩阵
print("Sparse matrix:\n",sparse_matrix)

# 将稀疏矩阵转换为密集矩阵
dense_matrix = sparse_matrix.todense()

# 打印密集矩阵
print("Dense matrix:\n", dense_matrix)

"""
在图卷积网络中的应用
在图卷积网络（GCN）中，节点特征通常表示为特征矩阵 X，其行对应于节点，列对应于特征。这个矩阵有时是稀疏的，以节省内存和计算资源。为了进行一些需要密集矩阵表示的操作，你可能需要将它转换为密集矩阵。例如，在图卷积层中，通常需要将特征矩阵与权重矩阵相乘，这时可以用 todense() 方法将特征矩阵从稀疏形式转换为密集形式：
"""

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor

# 假设features是一个稀疏矩阵
features = SparseTensor.from_dense(torch.randn(4, 4)) # 一个4x4的示例稀疏矩阵

# 将稀疏特征矩阵转换为密集矩阵
dense_features = features.to_dense()

# 使用密集矩阵进行一些操作。例如线性变化
weights = torch.randn(4, 2) # 示例权重矩阵
output = dense_features @ weights # 矩阵乘法

# 应用非线性激活函数
output = F.relu(output)
print(output)







