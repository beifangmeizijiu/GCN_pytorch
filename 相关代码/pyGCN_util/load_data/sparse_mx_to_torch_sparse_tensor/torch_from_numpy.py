"""
torch.from_numpy 是 PyTorch 中用于将 NumPy 数组转换为 PyTorch 张量的函数。这在需要在 NumPy 和 PyTorch 之间进行数据转换时非常有用。torch.from_numpy 创建的张量和原始的 NumPy 数组共享同一个内存空间，因此对其中一个的修改会影响另一个
用法：
语法：
torch.from_numpy(ndarray)
ndarray：要转换的 NumPy 数组。
返回值
返回一个与输入 NumPy 数组共享内存的 PyTorch 张量。
"""
import numpy as np
import torch

# 示例
# 以下示例展示了如何将 NumPy 数组转换为 PyTorch 张量，并说明了它们如何共享内存。
#
# 示例 1：基本用法
# 创建一个NumPy数组
np_array = np.array([1, 2, 3, 4, 5])

# 将NumPy数组转换为PyTorch张量
torch_tensor = torch.from_numpy(np_array)

print(torch_tensor)
print(torch_tensor.dtype)

# 示例 2：内存共享
# 修改Numpy数组
np_array[0] = 10
print(np_array)
print(torch_tensor)

# 修改PyTorch张量
torch_tensor[1] = 20
print(torch_tensor)
print(np_array)

# 示例 3：处理不同数据类型
# 如果你需要处理特定的数据类型，可以在创建 NumPy 数组时指定数据类型。torch.from_numpy 将保留这些类型。
# 创建一个浮点型 NumPy 数组
np_float_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

# 将 NumPy 数组转换为 PyTorch 张量
torch_float_tensor = torch.from_numpy(np_float_array)

print(torch_float_tensor)
print(torch_float_tensor.dtype)

"""
注意事项
共享内存：由于 torch.from_numpy 创建的张量与原始 NumPy 数组共享内存，因此它们之间的修改是互相影响的。这在某些情况下可能导致意外的行为，需特别注意。

数据类型支持：torch.from_numpy 支持大多数常见的 NumPy 数据类型，但并不是所有的类型都受支持。例如，不支持非数值类型的数组（如对象数组）。

不支持扩展：由于内存共享的特性，torch.from_numpy 转换后的张量不支持某些操作，例如改变形状（resize），因为这会涉及到内存的重新分配。

总结
torch.from_numpy 是一个高效且方便的函数，用于在 NumPy 和 PyTorch 之间进行数据转换。它的内存共享特性既是优点也是需要注意的地方，使得在 NumPy 和 PyTorch 之间切换和同步数据变得非常简便。理解和正确使用这一特性，可以在数据处理和机器学习建模中大大提升效率。
"""

"""
torch.from_numpy 和 torch.Tensor 是 PyTorch 中用于创建张量的两种方法，它们在用途和行为上有一些重要区别。

torch.from_numpy
功能
转换 NumPy 数组：将一个 NumPy 数组转换为 PyTorch 张量。转换后的张量与原始 NumPy 数组共享内存，这意味着对其中一个的修改会影响另一个。
注意事项
内存共享：转换后的张量与原始 NumPy 数组共享内存。因此，修改其中一个会影响另一个。
数据类型：torch.from_numpy 支持大多数 NumPy 数据类型，但并不是所有的类型都受支持。

torch.Tensor
功能
创建新的张量：直接创建一个 PyTorch 张量，可以指定数据和数据类型。torch.Tensor 是一种灵活的方法，用于创建各种初始化方式的张量。

"""
# 创建一个空的张量（未初始化）
tensor1 = torch.Tensor(5)

# 通过指定数据创建爱你张量
tensor2 = torch.Tensor([1, 2, 3, 4, 5])

print(tensor1)
print(tensor2)

"""
注意事项
不共享内存：通过 torch.Tensor 创建的张量与任何 NumPy 数组都不共享内存。它们是独立的。
数据类型默认是 float32：如果没有特别指定，torch.Tensor 创建的张量默认数据类型是 float32。
初始化问题：使用 torch.Tensor(size) 创建的张量没有进行初始化，可能包含任意数据。建议使用 torch.empty(size)、torch.zeros(size) 或其他初始化方法。
对比总结
内存共享：

torch.from_numpy：创建的张量与原始 NumPy 数组共享内存。
torch.Tensor：创建的张量与任何其他数据不共享内存。
数据来源：

torch.from_numpy：数据来自一个已有的 NumPy 数组。
torch.Tensor：数据可以来自一个 Python 列表、其他张量，或通过指定张量的形状创建未初始化的张量。
数据类型：

torch.from_numpy：保留 NumPy 数组的数据类型。
torch.Tensor：默认数据类型为 float32，可以通过其他函数创建指定数据类型的张量。
用法场景：

torch.from_numpy：适用于在 NumPy 和 PyTorch 之间共享数据时使用。
torch.Tensor：适用于直接在 PyTorch 中创建新的张量，可以初始化为特定的值或形状。
"""

# 示例对比

# 示例：torch.from_numpy
np_array = np.array([1, 2, 3, 4, 5])
torch_tensor_from_numpy = torch.from_numpy(np_array)
print("torch.from_numpy:", torch_tensor_from_numpy)

# 修改 NumPy 数组，查看张量的变化
np_array[0] = 10
print("Modified torch_tensor_from_numpy:", torch_tensor_from_numpy)

# 示例：torch.Tensor
torch_tensor = torch.Tensor(np_array)
print("torch.Tensor:", torch_tensor)

# 修改 NumPy 数组，不影响张量
np_array[0] = 20
print("Modified np_array:", np_array)
print("Unaffected torch_tensor:", torch_tensor)












