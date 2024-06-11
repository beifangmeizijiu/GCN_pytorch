"""
在 PyTorch 中，可以通过张量来索引另一个张量，这种操作被称为“高级索引”或“掩码索引”。这种方法允许你从一个张量中选择特定元素，或者根据条件从一个张量中提取子集。
"""
import torch
# 示例
# 1. 基本索引
# 你可以使用一个张量作为索引来选择另一个张量中的元素。例如：
# 创建一个张量
tensor = torch.tensor([10, 20, 30, 40, 50])

# 创建一个索引张量
indices = torch.tensor([0, 2, 4])

# 使用索引张量选择元素
selected_element = tensor[indices]
print(selected_element)

# 2. 布尔索引（掩码索引）
# 你也可以使用布尔张量作为索引来选择满足特定条件的元素。例如：

# 创建一个张量
tensor = torch.tensor([10, 20, 30, 40, 50])

# 创建一个布尔索引张量
mask = tensor > 25

# 使用布尔张量选择元素
selected_elements = tensor[mask]
print(mask)
print(selected_elements)  # 输出：tensor([30, 40, 50])

# 3。  多维索引
# 高级索引也可以用于多维张量。例如：
# 创建一个二维张量
tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

# 创建一个索引张量
row_indices = torch.tensor([0, 1, 2])
col_indices = torch.tensor([2, 1, 0])

# 使用索引张量选择元素
selected_elements = tensor[row_indices, col_indices]
print(selected_elements)  # 输出：tensor([3, 5, 7])


"""
总结
使用张量作为索引（tensor[tensor]）可以方便地从一个张量中选择特定元素或根据条件提取子集。高级索引和布尔索引是 PyTorch 中非常强大且灵活的功能，可以用来高效地操作和处理张量数据。
"""
