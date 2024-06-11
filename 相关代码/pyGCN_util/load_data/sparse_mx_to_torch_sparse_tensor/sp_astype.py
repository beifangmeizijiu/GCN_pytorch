"""
stype() 方法是 NumPy 和 SciPy 中用于转换数组或稀疏矩阵数据类型的一种常用方法。在 scipy.sparse 中，astype() 主要用于转换稀疏矩阵的数据类型，保持其稀疏结构不变。下面详细解释 astype() 方法的工作原理及其用法。

用法
astype() 方法可以应用于稀疏矩阵对象，将矩阵元素的数据类型转换为指定的类型。
语法
sparse_matrix.astype(dtype, casting='unsafe', copy=True)
dtype: 目标数据类型。例如，np.float32, np.float64, np.int32 等。
casting: 字符串，指定转换规则。常用值包括：
'no'：禁止强制转换。
'equiv'：只允许等价类型转换。
'safe'：只允许安全类型转换。
'same_kind'：允许相同种类的转换。
'unsafe'：允许任何转换（默认）。
copy: 布尔值，指定是否返回对象的副本。默认值为 True。
返回值
返回一个具有相同稀疏结构但元素类型为指定类型的新稀疏矩阵。

"""
import scipy.sparse as sp
import numpy as np
# 示例
# 以下示例展示了如何使用 astype() 方法将稀疏矩阵转换为不同的数据类型。
#
# 示例 1：基本用法
# 创建一个示例稀疏矩阵（CSR格式）
sparse_mx = sp.csr_matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

# 打印原始稀疏矩阵及其类型
print("Original sparse matrix:")
print(sparse_mx)
print("Type:",sparse_mx.dtype)

# 将稀疏矩阵的数据转化为 np.float32
sparse_mx_float32 = sparse_mx.astype(np.float32)

# 打印转换后的稀疏矩阵及其类型
print("\nConverted sparse matrix to float32:")
print(sparse_mx_float32)
print("Type:", sparse_mx_float32.dtype)

# 示例 2：使用不同的 dtype 和 casting 参数
# 将稀疏矩阵的数据类型转换为np.float64，安全转换
sparse_mx_float64_safe = sparse_mx.astype(np.float64, casting='safe')

print("\nConverted sparse matrix to float64 (safe casting):")
print(sparse_mx_float64_safe)
print("Type:", sparse_mx_float64_safe.dtype)


# 尝试不允许转换（这将引发错误）
try:
    sparse_mx_no_cast = sparse_mx.astype(np.float32)
except TypeError as e:
    print("\nError with casting='no':", e)

"""
astype() 方法是一个强大的工具，用于在保持稀疏矩阵结构不变的前提下转换其元素的数据类型。通过指定 dtype 和 casting 参数，你可以控制转换的行为和严格程度。这在处理不同精度需求的数据、优化计算性能和内存使用时非常有用。
"""









