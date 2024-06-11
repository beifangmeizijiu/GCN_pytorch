"""
torch.max 是 PyTorch 中用于计算张量中最大值的函数。它可以用于多个场景，例如找到张量的最大元素、沿指定维度查找最大值以及返回最大值对应的索引。以下是一些常见的用法和示例：
"""
import torch
# 用法
# 1、查找整个张量的最大值：

tensor = torch.tensor([1, 2, 3, 4, 5])
max_value = tensor.max()
print(max_value)

# 沿指定维度查找最大值
tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
# 沿第0维查找最大值（每列的最大值）
max_value, max_indices = torch.max(tensor, dim=0)
print(max_value)    # 输出：tensor([4, 5, 6])
print(max_indices)  # 输出：tensor([1, 1, 1])

# 等同写法
max_value, max_indices = tensor.max(0)
print(max_value)
print(max_indices)

# 沿第1维查找最大值（每行的最大值）
max_value, max_indices = torch.max(tensor, dim=1)
print(max_value)    # 输出：tensor([3, 6])
print(max_indices)  # 输出：tensor([2, 2])

"""
返回最大值及其索引：

torch.max 返回两个张量：最大值张量和最大值所在位置的索引张量。这在分类任务中特别有用，例如获取预测类别的索引。
"""

# 综合示例
# 以下是一个分类任务中常用的示例，假设模型输出是每个类别的分数（logits），我们需要找到每个样本的预测类别：

# 模拟批量大小为3，类别数为4的输出 (logits)
output = torch.tensor([[1.2, 0.5, -0.8, 2.4],
                       [0.3, 1.8, 0.9, -0.5],
                       [-1.1, 2.2, 0.1, 0.9]])

# 使用 torch.max 找到每个样本的最大值和对应的类别索引
max_value, preds = torch.max(output, dim=1)
print("Max values:\n", max_value)  # 输出每个样本的最大值
print("Predicted classes:\n", preds)  # 输出每个样本的预测类别索引

"""
总结
torch.max 是一个非常有用的函数，可以帮助我们在张量中查找最大值及其索引，特别是在处理分类任务时，它可以帮助我们获取模型的预测结果。通过指定维度参数 dim，我们可以灵活地在不同的维度上进行操作，获取所需的最大值和索引信息。
"""