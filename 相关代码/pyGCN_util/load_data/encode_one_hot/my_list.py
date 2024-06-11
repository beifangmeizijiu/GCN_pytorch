"""
列表推导式（List Comprehensions）是 Python 中一种简洁、直观的创建列表的方法。它允许你用一行代码来生成一个列表，可以包含条件和嵌套循环，使代码更加简洁和易读。
基本语法
[expression for item in iterable if condition]

expression 是每个列表元素的计算方式，可以是任意有效的 Python 表达式。
item 是从 iterable 中取出的元素。
iterable 是一个可迭代对象，比如列表、元组、字符串等。
condition 是一个可选的条件，用于筛选元素。
"""

# 示例
# 1. 创建一个简单的列表
# 创建一个包含 0 到 9 的列表：
numbers = [x for x in range(10)]
print(numbers)

# 2. 创建一个平方数列表
# 创建一个包含 0 到 9 的平方的列表：
squares = [x**2 for x in range(10)]
print(squares)

# 3. 使用条件过滤
# 创建一个包含 0 到 9 中的偶数的列表：
evens = [x for x in range(10) if x%2 == 0]
print(evens)

# 4. 嵌套循环
# 创建一个包含所有可能的 (x, y) 组合的列表，其中 x 来自于 [0, 1, 2]，y 来自于 [0, 1, 2]：
combinations = [(x,y) for x in range(3) for y in range(3)]
print(combinations)

# . 条件和嵌套循环结合
# 创建一个包含所有可能的 (x, y) 组合的列表，但只保留 x 和 y 不相等的组合：
combinations = [(x,y) for x in range(3) for y in range(3) if x!= y]
print(combinations)

"""
列表推导式的优点
简洁：通常比传统的循环和条件语句更简洁。
易读：如果使用得当，列表推导式的意图通常非常明确，易于理解。
效率：由于 Python 内部对列表推导式进行了优化，通常比传统循环效率更高。
列表推导式的缺点
可读性：对于非常复杂的表达式，列表推导式可能会变得难以阅读和理解。在这种情况下，使用传统的循环和条件语句可能会更清晰。
调试困难：由于列表推导式是一行代码，如果其中包含错误，调试可能会比较困难。
总结
列表推导式是 Python 中一种非常强大的工具，可以简化代码，提高可读性和效率。合理使用列表推导式可以使代码更加简洁和清晰，但在遇到复杂逻辑时，仍需平衡可读性和简洁性。
"""






