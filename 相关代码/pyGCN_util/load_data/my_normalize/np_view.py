"""
NumPy 中的 view 方法。它可以帮助我们创建新的数组对象，而不需要复制数据。
"""
import numpy as np

# 1. 改变数据类型
# 使用 view 可以创建一个具有不同数据类型的新数组，但它与原数组共享相同的数据存储。
# 使用 view 方法改变数据类型时，它只是改变了数据的解释方式，而不修改数据的实际存储内容。这意味着新视图只是对底层相同的二进制数据进行了不同的解读。

# 创建一个整数类型的数组
a = np.array([1, 2, 3, 4], dtype=np.int32)

# 查看数组 a 的数据类型
# 这个数组在内存中的表示是：[0x00000001, 0x00000002, 0x00000003, 0x00000004]
print("Original array:", a)
print("Original dtype:", a.dtype)

# 使用 view 创建一个新的浮点类型视图
# 数组 a 仍然是 [1, 2, 3, 4]，数据类型为 int32，而视图 b 对应的解释变为 float32，内容看起来像随机的很小的浮点数，因为这些数值其实是以浮点格式解释的整数。
b = a.view(np.float32)

# 查看新数组 b 的数据类型
print("New view array:", b)
print("New view dtype:", b.dtype)

b[0] = 0.5

# 查看原数组和视图的变化
print("Modified view array:", b)
print("Original array after modification:", np.float32(a[0]))

"""
使用 view 方法创建的新数组对象只改变数据的解释方式，而不改变数据的实际存储。
对视图进行修改会影响到原数组，因为它们共享相同的数据存储。
这种方法适用于需要在不同数据类型之间切换，而无需复制数据的情况。
"""

# 2. 改变形状
# 使用 view 可以在不复制数据的情况下查看数组的不同形状。

# 创建一个 1D 数组
a = np.array([1, 2, 3, 4, 5, 6])

b = a.view().reshape(2, 3)

# 查看原数组和视图的形状
print("Original array:", a)
print("Original shape:", a.shape)
print("New view array:\n", b)
print("New view shape:", b.shape)

# 修改视图中的值
b[0, 0] = 10

# 查看原数组和视图的变化
print("Modified view array:\n", b)
print("Original array after modification:", a)

# 3.更改数组类型但保持形状不变
# 通过 view 方法，还可以更改数组的类型，但保持其形状不变。

# 创建一个整数类型的数组
a = np.array([1, 2, 3, 4], dtype=np.int8)
# 使用 view 创建一个新的无符号整数类型视图
b = a.view(np.uint8)

# 查看原数组和视图的数据类型和内容
print("Original array:", a)
print("Original dtype:", a.dtype)
print("New view array:", b)
print("New view dtype:", b.dtype)


# 修改视图中的值
b[0] = 255

# 查看原数组和视图的变化
print("Modified view array:", b)
print("Original array after modification:", a)

"""
在以上示例中，通过 view 方法，我们能够创建一个新数组对象，该对象共享与原数组相同的数据存储。通过这种方式，我们可以查看相同数据的不同表示形式，而无需进行数据复制
"""












