"""
map 函数是 Python 内置的高阶函数，用于将一个函数应用到一个或多个可迭代对象的所有元素上，并返回一个迭代器。map 函数的基本语法如下：
map(function, iterable, ...)
function：要应用的函数。
iterable：一个或多个可迭代对象（如列表、元组、字符串等）。
"""
# 示例 下面是一些 map 函数的示例，帮助理解其用法。
# 示例 1：对列表中的每个元素进行平方计算
def square(x):
    return x*x

numbers = [1, 2, 3, 4, 5]
squared_numbers = map(square, numbers)

# map 返回的是一个迭代器，需要将其转换为列表以便查看结果
print(list(squared_numbers))

# 示例 2：将两个列表中的元素相加

list1 = [1, 2, 3]
list2 = [4, 5, 6]
summed_list = map(lambda x, y:  x +y, list1,list2)
print(list(summed_list))

# 示例 3：将字符串中的每个字符转换为其 ASCII 值
string = 'hello'
ascii_values = map(lambda char: ord(char), string)
print(list(ascii_values))

# 示例 4：将列表中的每个元素转换为字符串
numbers = [1, 2, 3, 4, 5]
str_numbers = map(str, numbers)
print(list(numbers))

"""
注意事项
map 函数返回的是一个迭代器，在需要输出时可以使用 list() 或 tuple() 将其转换为列表或元组。
map 函数可以接受多个可迭代对象，这些对象的长度应相同，如果长度不同，则按最短的可迭代对象进行计算。
map 函数是惰性计算的，即只有在迭代结果时才会进行计算。
"""

