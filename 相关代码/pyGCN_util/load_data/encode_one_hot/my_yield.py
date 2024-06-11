"""
yield 是 Python 中用于生成器的关键字。生成器是一种特殊的迭代器，通过 yield 关键字实现，可以在迭代过程中保存函数的运行状态，生成一系列的值。

生成器与 yield
生成器函数：包含一个或多个 yield 语句的函数。调用生成器函数并不会立即执行它的代码，而是返回一个生成器对象。
生成器对象：可以用在 for 循环中迭代，或用 next() 函数手动获取下一个值。
yield 语句
作用：在生成器函数中，yield 暂停函数的执行并返回一个值。函数在下次被调用时从 yield 语句之后继续执行。
语法：yield <expression>

"""
# 示例
# 基本示例
def simple_generator():
    yield 1
    yield 2
    yield 3

gen = simple_generator()

print(next(gen))  # 输出: 1
print(next(gen))  # 输出: 2
print(next(gen))  # 输出: 3

# 使用 for 循环迭代生成器

for value in simple_generator():
    print(value)

"""
yield 和普通函数的区别
普通函数：一次性执行完毕，返回一个值。
生成器函数：每次调用 yield 生成一个值，函数的状态会被保存，下一次从断点处继续执行。
使用场景
处理大量数据：生成器一次生成一个值，不会一次性将所有值加载到内存中，适合处理大数据。
延迟计算：生成器在需要时生成值，而不是立即计算所有值。
流水线处理：生成器可以与其他生成器或迭代器结合，实现数据的逐步处理
"""
# 示例：生成斐波那契数列

def fibonacci(n):
    a,b =0,1
    for _ in range(n):
        yield a
        a,b = b,a+b

print("生成兔子数列")

for num in fibonacci(10):
    print(num)

# 示例：生成无限数列
def infinite_sequence():
    num = 0
    while True:
        yield num
        num += 1

gen = infinite_sequence()
print("无限数列")
for _ in range(10):
    print(next(gen))

# 总结
# yield 是生成器函数的核心，用于生成值并保存函数状态。
# 生成器在处理大量数据和延迟计算时非常有用。
# 生成器表达式是一种简洁的生成器创建方式。
# 通过 yield，Python 允许我们创建高效、内存友好的代码，特别是在需要逐步处理数据时。







