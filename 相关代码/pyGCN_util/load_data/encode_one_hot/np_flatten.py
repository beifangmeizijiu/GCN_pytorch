"""
flatten 是 NumPy 中的一个方法，用于将多维数组转换为一维数组。它返回的是数组的副本，不会修改原始数组。你可以指定 order 参数来控制数据在内存中的存储顺序。
语法
array.flatten(order='C')
参数
order：指定展开数组的顺序，默认是 'C'（行优先）。可以选择：
'C'：行优先（C 风格）
'F'：列优先（Fortran 风格）
'A'：如果原数组是行优先的就使用行优先，如果是列优先的就使用列优先
'K'：按照数组在内存中的布局顺序展开
"""
import numpy as np
# 示例 1：使用默认的行优先（C 风格）展开
arr = np.array([[1, 2, 3], [3, 4, 5]])
flat_arr = arr.flatten()

print("原始数组：")
print(arr)

print("\n展开后的数组（行优先）：")
print(flat_arr)

# 示例 2：使用列优先（Fortran 风格）展开
flat_arr_f = arr.flatten(order='F')

print("原始数组：")
print(arr)

print("\n展开后的数组（列优先）：")
print(flat_arr_f)

"""
注意
flatten() 返回的是数组的副本，如果对返回的数组进行修改，不会影响原始数组。
如果你希望对原始数组进行修改，可以使用 ravel() 方法，它会返回数组的视图（如果可能的话）。
"""
# 示例 3：使用 ravel() 展开数组

flat_arr_ravel = arr.ravel(order='F')

print("原始数组：")
print(arr)

print("\n使用 ravel 展开后的数组（列优先）：")
print(flat_arr_ravel)
"""
ravel() 和 flatten() 的主要区别在于 ravel() 返回的是数组的视图，这意味着对 ravel() 返回的数组进行修改会影响到原始数组。
"""










