"""
np.power 是 NumPy 库中的一个函数，用于逐元素地计算数组的指数（幂）。它接受两个数组或一个数组和一个标量，返回一个新数组，其中每个元素是第一个数组中对应元素的幂。
函数签名
numpy.power(x1, x2, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True)

参数
x1: 第一个输入数组，表示底数。
x2: 第二个输入数组，表示指数。可以是与 x1 形状相同的数组或一个标量。
out: 可选。用于存放计算结果的数组。其形状必须与输入数组相同。
where: 可选。数组或条件，用于指定要计算的元素。
casting: 可选。用于指定输入数组到输出数组的强制转换规则。
order: 可选。指定输出数组的存储顺序。
dtype: 可选。指定输出数组的数据类型。
subok: 可选。指定输出数组是否使用与输入相同的子类。
"""
import numpy as np

# 示例代码
# 基本用法

# 创建两个数组
a = np.array([1, 2, 3, 4])
b = np.array([1, 2, 3, 4])

# 逐元素计算 a 的 b 次方
result = np.power(a,b)
print(result)

# 使用标量指数

# 创建一个数组
a = np.array([1, 2, 3, 4])

# 使用标量指数
result = np.power(a, 2)
print(result)

# 使用 out 参数

# 创建两个数组
a = np.array([1, 2, 3, 4])
b = np.array([1, 2, 3, 4])


# 创建一个与输入数组形状相同的输出数组
out = np.empty_like(a)

# 逐元素计算 a 的 b 次方，并将结果存储在 out 数组中
np.power(a, b, out=out)
print(out)

# 广播机制
# 通过广播机制，对数组的每个元素计算相同的幂
base = np.array([1, 2, 3, 4])
exp = np.array([2])
result = np.power(base, exp)
print("广播机制")
print(result)

# 广播机制的例子
# 广播机制允许 NumPy 在计算时自动扩展数组的形状，使其兼容。例如：
# base 是一个2x2数组
base = np.array([[1, 2], [3, 4]])
# exp 是一个标量
exp = 2
result = np.power(base,exp)
print(result)

# where 参数
# where 参数用于指定计算条件，它是一个布尔数组，与输入数组形状相同。只有当 where 中对应位置为 True 时，才会对该位置进行计算
"""
原理一个是把两个数组做指数运算，然后给放到一个新的numpy数组里面（然后如果不符合where条件则跳过，这应该是底层实现）；很不巧的是新数组他不初始化，所以他的答案不确定
"""

print(np.__version__)
# 创建两个数组
a = np.array([1, 2, 3, 4])
b = np.array([1, 2, 3, 4])
result = np.power(a,b, where=(a%2 == 0))
print(result)

out = np.array(np.ones_like(a))
c = np.array([False, True, False, True])

# 使用条件来计算部分元素的幂
np.power(a, b, out=out, where=c)

print(out)

out = np.array(np.zeros_like(a))

# 使用条件来计算部分元素的幂
np.power(a, b, out=out, where=c)


print(out)

# subok 参数
# subok 参数用于控制返回的数组是否保留子类。如果为 True，返回的数组将保留子类；如果为 False，返回的数组将强制转换为基类。
class MyArray(np.ndarray):
    pass

# 创建 MyArray 的实例
x = np.array([1, 2, 3]).view(MyArray)

# 使用 subok=True，保留子类
result_subok_true = np.power(x, 2, subok=True)
print(type(result_subok_true))

# 使用 subok=False，转换为基类
result_subok_false = np.power(x, 2, subok=False)
print(type(result_subok_false))

# casting 参数
# casting 参数用于控制数据类型转换的规则，以下是几个可选值：
# 'no': 不允许任何类型转换。
# 'equiv': 只允许等价类型转换。
# 'safe': 只允许安全的转换，不允许数据丢失或精度下降。
# 'same_kind': 允许安全转换和等价转换。
# 'unsafe': 允许所有类型转换。

# 创建一个整型数组
x1 = np.array([1, 2, 3], dtype=np.int32)
x2 = np.array([2, 2, 2], dtype=np.float64)

# 使用 casting='safe'，允许安全转换
result_safe = np.power(x1, x2, casting='safe')
print(result_safe)

# 使用 casting='unsafe'，允许所有转换
result_unsafe = np.power(x1, x2, casting = 'unsafe')
print(result_unsafe)

# 使用 casting='no'，不允许任何转换（会报错）
try:
    result_no = np.power(x1, x2, casting = 'no')
except TypeError as e:
    print(e)

"""
总结
where 参数：指定计算条件。
subok 参数：控制返回的数组是否保留子类。
casting 参数：控制数据类型转换的规则。
这些参数提供了更灵活和安全的控制，使得 np.power 函数在处理不同类型和条件的数组时更加高效和可靠。
"""











