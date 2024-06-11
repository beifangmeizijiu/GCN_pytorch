"""
zip 方法在 Python 中用于将多个可迭代对象（如列表、元组、字符串等）聚合成一个迭代器。这个迭代器会生成一系列的元组，其中的第 i 个元组包含了所有输入可迭代对象的第 i 个元素。
"""

# 基本用法
# 创建两个列表
list1 = [1, 2, 3]
list2 = ['a', 'b', 'c']

# 使用zip函数
ziped = zip(list1,list2)

# 将zip对象转换为列表
result = list(ziped)

print(result)

# 特性和注意事项
# 1、输入长度不同时： zip 会在最短的输入可迭代对象到达末尾时停止。
list1 = [1, 2, 3]
list2 = ['a', 'b']

result = list(zip(list1,list2))
print(result)

# 解压缩： 可以使用 zip 的反向操作来解压缩序列。
pairs = [(1, 'a'), (2, "b"), (3, 'c')]

# 使用zip(*iterable)来解压缩
list1, list2 =zip(*pairs)

print(list1)
print(list2)

# 实际例子
# 1、并行迭代： 使用 zip 可以在循环中并行迭代多个可迭代对象。
names = ['Alice', 'Bob', 'Charlie']
scores = [85, 90, 78]

for name, score in zip(names, scores):
    print(f"{name} scored {score}")

# 2、构建字典： 可以用 zip 来将两个列表合并成一个字典。
keys = ['name', 'age', 'city']
values = ['Alice', 25, 'New York']

my_dict = dict(zip(keys, values))

print(my_dict)  # 输出: {'name': 'Alice', 'age': 25, 'city': 'New York'}

# 3、矩阵转置

matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

transposed = list(zip(*matrix))

print(transposed)

