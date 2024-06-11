"""
np.random.normal 是 NumPy 中生成正态（高斯）分布随机数的函数。它返回一个指定均值和标准差的正态分布随机数数组。
np.random.normal(loc=0.0, scale=1.0, size=None)

参数
loc：float，正态分布的均值（默认值为0.0）。
scale：float，正态分布的标准差（必须是非负的，默认值为1.0）。
size：int 或者 tuple of ints，输出数组的形状。如果是 None（默认值），则返回一个标量。
返回值
返回指定均值和标准差的正态分布随机数数组。
"""
import numpy as np
# 示例
# 生成一个标量

# 生成一个均值为0，标准差为1的正态分布随机数
scalar = np.random.normal()
print(scalar)

# 生成一个数组
# 生成一个均值为0，标准差为1的正态分布随机数数组，形状为(3, 5)
array = np.random.normal(0.0, 1.0, (3, 5))
print(array)






