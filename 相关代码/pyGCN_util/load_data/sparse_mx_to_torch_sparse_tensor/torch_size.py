"""
torch.Size 是 PyTorch 中用于表示张量形状的对象。它是一个不可变的元组，提供了一些有用的方法来处理张量的形状信息。在 PyTorch 中，张量的形状用 torch.Size 对象表示，这使得你可以方便地查询和操作张量的维度。
"""
import torch
# 基本用法
# 创建 torch.Size 对象
# torch.Size 通常由 PyTorch 自动创建并附加到张量对象上。你可以通过张量的 .size() 方法来获取它的形状。
# 创建一个张量
tensor = torch.randn(2, 3, 4)

# 获取张量的形状
size = tensor.size()

print(tensor)
print(size)
print(type(size))

# 使用 torch.Size 直接创建
# 你也可以直接创建一个 torch.Size 对象。
size = torch.Size([2, 3, 4])
print(size)

# 属性和方法
# 属性
# torch.Size 本质上是一个元组，因此你可以使用元组的所有属性和方法。
# 方法
# torch.Size 具有元组的所有方法，因为它继承自元组。以下是一些常用方法和操作：
#
# 索引和切片：

print(size[0])
print(size[:2])

# 长度：
print(len(size))

# 迭代
for dim in size:
    print(dim)

# 与其他元组或列表进行操作：

# 转换为列表
size_list = list(size)
print(size_list)  # [2, 3, 4]

# 连接其他元组
new_size = size + (5,)
print(new_size)  # torch.Size([2, 3, 4, 5])

# 示例
# 改变张量形状
# 你可以使用 torch.Size 对象来改变张量的形状。
reshaped_tensor = tensor.view(size[0], -1)
print(reshaped_tensor.size())

# 动态调整张量形状
# 动态调整张量形状时，torch.Size 可以非常方便地帮助你实现这一操作。
def reshape_tensor(tensor, new_shape):
    return tensor.view(torch.Size(new_shape))

new_shape = [6, 4]
reshaped_tensor = reshape_tensor(tensor, new_shape)
print(reshaped_tensor.size())

"""
总结
torch.Size 是 PyTorch 中用于表示张量形状的一个重要对象。它继承了元组的所有属性和方法，使得操作张量形状变得简单和直观。理解和利用 torch.Size 对象，可以帮助你在处理张量时更加灵活和高效。
"""