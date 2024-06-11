"""
filter 函数用于从一个序列中过滤出符合特定条件的元素。它接受一个函数和一个可迭代对象作为参数，并返回一个迭代器，该迭代器只包含那些使传入函数返回 True 的输入可迭代对象中的元素。换句话说，返回的迭代器中包含的元素是那些通过了过滤条件的元素。
语法
filter(function, iterable)
function: 一个函数，该函数接受一个参数并返回布尔值 True 或 False。
iterable: 一个可迭代对象，如列表、元组、集合等。
"""
# 示例
# 1.使用 filter 过滤列表中的偶数
# 定义一个函数，判断一个数是否是偶数
def is_even(n):
    return n % 2 == 0

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 使用 filter 过滤出偶数
even_numbers = filter(is_even, numbers)

# 将结果转换为列表并打印
print(list(even_numbers))

"""
在这个示例中，is_even 函数返回 True 的元素（即偶数）将包含在 even_numbers 迭代器中。然后我们将迭代器转换为列表，以查看其内容。
"""
# 2.使用 lambda 函数和 filter
# 使用 lambda 函数和 filter 过滤出偶数
even_numbers = filter(lambda n: n % 2 == 0, numbers)

# 将结果转换为列表并打印
print(list(even_numbers))

# 3.使用 lambda 函数和 filter

words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]

# 使用 lambda 函数和 filter 过滤出长度大于5的单词
long_words = filter(lambda word: len(word) > 5, words)

# 将结果转换为列表并打印
print(list(long_words))

# 4.结合 filter 和 map 使用
# 有时你可能想先过滤数据，然后对过滤后的数据进行处理。可以将 filter 和 map 结合使用。

even_squares = map(lambda n: n ** 2, filter(lambda n:n%2 ==0 ,numbers))

# 将结果转换为列表并打印
print(list(even_squares))

"""
总结
filter 函数从一个可迭代对象中过滤出符合条件的元素，并返回一个只包含这些元素的迭代器。
filter 返回的迭代器需要转换为列表或其他可迭代对象以查看结果。
filter 函数通常与 lambda 函数结合使用，以实现简洁的过滤条件。
"""

"""
filter直接筛选元素有什么区别
"""
# 1. 使用 filter 函数
# filter 函数接受一个函数和一个可迭代对象，返回一个迭代器。迭代器只包含使函数返回 True 的元素。
# 示例
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 使用 filter 函数过滤出偶数
even_numbers = filter(lambda n: n % 2 == 0, numbers)

# 将结果转换为列表并打印
print(list(even_numbers))  # 输出: [2, 4, 6, 8, 10]
# 2. 使用列表推导式
# 列表推导式是一种紧凑的方式来创建列表，其中包含满足特定条件的元素。
# 示例
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 使用列表推导式过滤出偶数
even_numbers = [n for n in numbers if n%2 ==0]
print(even_numbers)

# 3. 使用循环
# 直接使用循环和条件语句来筛选元素。
#
# 示例
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 使用循环和条件语句过滤出偶数
even_numbers = []
for n in numbers:
    if n % 2 == 0:
        even_numbers.append(n)

print(even_numbers)  # 输出: [2, 4, 6, 8, 10]

"""
比较
可读性和简洁性
filter 函数: 更加简洁，表达功能直接，特别是对于熟悉函数式编程的人来说更为自然。
列表推导式: 通常更容易理解和阅读，因为它直接在列表的构造中包含了逻辑。
循环和条件语句: 更加冗长，但对于不熟悉列表推导式或函数式编程的人来说，可能更容易理解。

性能
filter 函数: 返回一个迭代器，惰性求值，只有在迭代时才会计算元素，适合处理大数据集。
列表推导式: 立即求值，创建一个新的列表，适合处理中小型数据集。
循环和条件语句: 与列表推导式类似，立即求值，但更加冗长。

适用场景
filter 函数: 适合函数式编程风格的代码，或者需要处理大数据集而希望延迟计算的场景。
列表推导式: 适合简洁表达筛选逻辑的场景，通常在处理中小型数据集时使用。
循环和条件语句: 适合需要更多控制或需要在筛选过程中执行复杂逻辑的场景。

总结
filter 函数适用于需要简洁表达筛选逻辑并且习惯函数式编程的场景。
列表推导式适用于需要直接创建列表并且希望代码更加易读的场景。
循环和条件语句适用于需要更多控制或者不熟悉上述两种方法的场景。
具体选择哪种方法，取决于代码的需求和开发者的偏好。

"""

































