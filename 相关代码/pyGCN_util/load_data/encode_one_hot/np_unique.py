"""
np.unique(labels) 是 NumPy 中的一个函数，用于查找数组中的唯一元素，并返回它们的有序列表。除此之外，它还可以返回唯一元素在原数组中的索引以及唯一元素的出现次数。
"""
import numpy as np
# 参数
# labels：输入的数组，可以是任何维度的数组。
# 返回值
# 返回一个包含唯一元素的数组，元素按升序排列。
labels = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
unique_labels = np.unique(labels)
print(unique_labels)

"""
其他功能
np.unique 函数还有两个可选参数 return_index 和 return_counts，可以用来获取唯一元素在原数组中的索引以及它们的出现次数。
"""
# 示例：获取唯一元素及其在原数组中的索引
unique_labels, indices = np.unique(labels, return_index=True)
print("Unique labels:", unique_labels)
print("Indices:", indices)
# 示例获取唯一元素及其出现次数
unique_labels, counts = np.unique(labels, return_counts=True)
print("Unique labels:", unique_labels)
print("Counts:", counts)
# 示例结合 return_index 和 return_counts
unique_labels, indices, counts = np.unique(labels, return_index=True, return_counts=True)
print("Unique labels:", unique_labels)
print("Indices:", indices)
print("Counts:", counts)
"""
应用场景
去重：可以用来去除数组中的重复元素。
统计：统计元素的出现频率。
索引：获取唯一元素在原数组中的首次出现位置。
"""


