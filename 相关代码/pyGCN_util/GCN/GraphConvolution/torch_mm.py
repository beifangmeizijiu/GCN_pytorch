"""
在 PyTorch 中，使用 @ 运算符进行矩阵乘法与使用 torch.mm 或 torch.matmul 的效果是相同的，但它们之间有一些细微的差别，尤其是在处理不同维度的张量时。

"""
import torch
# @ 运算符
# 在 Python 中，@ 运算符用于矩阵乘法。在 PyTorch 中，它相当于 torch.matmul，可以处理任意维度的张量。
# 示例
# 创建两个二维张量（矩阵）
matrix1 = torch.tensor([[1, 2], [3, 4]])
matrix2 = torch.tensor([[5, 6], [7, 8]])

# 使用@运算符进行矩阵乘法
result = matrix1 @ matrix2
print(result)

# torch.mm
# torch.mm 仅适用于二维张量（矩阵）的乘法。如果你尝试对更高维度的张量使用 torch.mm，会报错。
# 示例
# 使用 torch.mm 进行矩阵乘法
result = torch.mm(matrix1,matrix2)
print(result)

# torch.matmul
# torch.matmul 更加通用，可以处理任意维度的张量。
# 示例
# 使用 torch.matmul 进行矩阵乘法
tensor1 = torch.randn(2, 3, 4)
tensor2 = torch.randn(2, 4, 5)
# 使用 torch.matmul 进行张量乘法
result = torch.matmul(tensor1,tensor2)
print(result.shape)

# 区别总结
# 1、适用范围：
#
# @ 运算符和 torch.matmul 都可以用于任意维度的张量，并处理高维张量的批量矩阵乘法。
# torch.mm 仅适用于二维张量（矩阵）。
# 2、使用方便性：
#
# @ 运算符是 Python 3.5+ 的内置操作符，用于矩阵乘法，语法上更加简洁直观。
# torch.matmul 是 PyTorch 提供的函数，功能强大且通用。
# 3、行为：
#
# 对于二维张量（矩阵），@ 运算符、torch.mm 和 torch.matmul 的行为是相同的。
# 对于高维张量，@ 运算符和 torch.matmul 支持批量矩阵乘法，并自动处理张量的广播。

# 示例对比
# 二维张量（矩阵）
matrix1 = torch.tensor([[1, 2], [3, 4]])
matrix2 = torch.tensor([[5, 6], [7, 8]])
# 使用@运算符
result_at = matrix1 @ matrix2
print("Result using @:", result_at)

# 使用 torch.mm
result_mm = torch.mm(matrix1, matrix2)
print("Result using torch.mm", result_mm)

# 使用torch.matmul
result_matmul = torch.matmul(matrix1, matrix2)
print("Result using torch.matmul:", result_matmul)

# 高维张量

tensor1 = torch.randn(2, 3, 4)
tensor2 = torch.randn(2, 4, 5)

# 使用@运算符
result_at = matrix1 @ matrix2
print("Result using @:", result_at)

# 使用torch.matmul
result_matmul = torch.matmul(matrix1, matrix2)
print("Result using torch.matmul:", result_matmul)

# 使用 torch.mm 会报错
try:
    result_mm = torch.mm(tensor1, tensor2)
except RuntimeError as e:
    print("Error using torch.mm:", e)




