"""
lambda 函数是 Python 中的一种匿名函数，它可以用来定义简单的、一次性的函数。它们与常规的 def 函数不同，因为它们没有名字，并且只能包含一个表达式。语法为：
lambda arguments: expression

"""
# 示例：单个返回值的 lambda 函数
# 一个简单的 lambda 函数，计算输入数值的平方：
square = lambda x: x**2
print(square(4))

# 示例：多个参数的 lambda 函数
# 一个 lambda 函数，接受两个参数并返回它们的和：
add = lambda x, y: x + y
print(add(3, 5))

# 示例：在 map 中使用 lambda 函数
# 将一个列表中的每个元素平方：
numbers = [1, 2, 3, 4, 5]
squared_numbers = map(lambda x: x ** 2, numbers)
print(list(squared_numbers))  # 输出：[1, 4, 9, 16, 25]

# 示例：返回多个值的 lambda 函数
# 尽管 lambda 函数只能包含一个表达式，但你可以通过返回元组(字符串，列表都可以)的方式实现返回多个值：
multiple_returns = lambda x: (x, x*2 ,x **2)
print(multiple_returns(3))

# 示例：结合 map 使用 lambda 返回多个值
# 对列表中的每个元素进行多次变换并返回多个值：
numbers = [1, 2, 3, 4, 5]

# 定义一个 lambda 表达式返回两个值
multiple_transforms = map(lambda x: (x + 1, x * 2), numbers)
results = list(multiple_transforms)

for res in results:
    print(f"Original + 1: {res[0]}, Original * 2: {res[1]}")

# 示例：结合 filter 和 lambda
# 使用 lambda 和 filter 函数过滤列表中的偶数：
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = filter(lambda x : x%2 ==0, numbers)
print(list(even_numbers))

# 示例：结合 reduce 和 lambda
# 使用 lambda 和 reduce 计算列表元素的乘积：
from functools import reduce

numbers = [1, 2, 3, 4, 5]
product = reduce(lambda x,y: x * y, numbers)
print(product)













