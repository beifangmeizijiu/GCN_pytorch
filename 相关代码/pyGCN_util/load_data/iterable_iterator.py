"""
迭代器和可迭代对象在 Python 中是两个紧密相关但有显著区别的概念。理解它们之间的区别对于掌握 Python 的迭代机制至关重要。
"""

"""
可迭代对象 (Iterable)
定义：
可迭代对象是指实现了 __iter__ 方法并返回一个迭代器的对象，或者实现了 __getitem__ 方法并能够从索引 0 开始依次取出元素的对象。

特点：

可以用 for 循环进行迭代。
包含序列（如列表、元组、字符串）和某些非序列类型（如字典、集合）。
"""
# 示例

# 列表是一个可迭代对象
my_list = [1, 2, 3]

# 字符串是一个可迭代对象
my_string = "hello"

# 字典是一个可迭代对象
my_dict = {'a': 1, 'b': 2}

# 可以用 for 循环迭代这些可迭代对象
for element in my_list:
    print(element)

for char in my_string:
    print(char)

for key in my_dict:
    print(key)

"""
迭代器 (Iterator)
定义：
迭代器是一个实现了 __iter__ 和 __next__ 方法的对象。__iter__ 方法返回迭代器本身，__next__ 方法返回序列中的下一个元素，如果没有更多元素时会引发 StopIteration 异常。

特点：

迭代器是一次性的，不能重新开始迭代。
每次调用 __next__ 方法时返回下一个元素，直到没有更多元素为止。
更节省内存，因为它们不是将所有元素存储在内存中，而是逐个生成元素。
"""
# 通过 iter() 将列表转换为迭代器
my_list = [1, 2, 3]
my_iterator = iter(my_list)

# 使用 next() 迭代元素
print(next(my_iterator))  # 输出: 1
print(next(my_iterator))  # 输出: 2
print(next(my_iterator))  # 输出: 3

# 当没有更多元素时，引发 StopIteration 异常
# print(next(my_iterator))  # 会引发 StopIteration 异常

"""
关系与区别
关系：
可迭代对象实现了 __iter__ 方法。
迭代器实现了 __iter__ 和 __next__ 方法。
调用 iter() 函数可以从一个可迭代对象中获取一个迭代器。
区别：
可迭代对象：能返回一个迭代器的对象，如列表、元组、字典、集合、字符串等。它们实现了 __iter__ 方法。
迭代器：用来实际执行迭代过程的对象。它实现了 __iter__ 和 __next__ 方法。
"""

# 例子说明区别
# 列表是一个可迭代对象
my_list = [1, 2, 3]

# 获取一个迭代器
my_iterator = iter(my_list)

# 使用迭代器迭代
print(next(my_iterator))  # 输出: 1
print(next(my_iterator))  # 输出: 2

# 列表依然可以被重新迭代
for element in my_list:
    print(element)

# 迭代器不能被重新迭代
# 会继续从上次停止的地方开始迭代
print(next(my_iterator))  # 输出: 3
# print(next(my_iterator))  # 会引发 StopIteration 异常

"""
总之，可迭代对象是能够返回迭代器的对象，而迭代器则是实际执行迭代过程的对象。可迭代对象可以被多次迭代，而迭代器一旦用尽，就不能再重新开始迭代。
"""










