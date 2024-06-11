"""
生成器表达式在 Python 中用于创建生成器对象，它是使用圆括号定义的，类似于列表推导式，但生成器表达式是按需生成项的，这使得它比列表推导式更节省内存。生成器表达式的语法如下：
(expression for item in iterable if condition)

下面是一些使用生成器表达式的示例：
"""
# 基本用法
# 创建一个生成器表达式，它生成 0 到 9 的平方：
gen_exp = (x**2 for x in range(10))


# 使用生成器
for num in gen_exp:
    print(num)

# 转换为列表、元组、集合
# 虽然生成器表达式本身不会创建列表、元组或集合，但可以将生成器表达式的结果传递给相应的构造函数来创建这些对象：
# 转换为列表
gen_exp = (x**2 for x in range(10))
list_result = list(gen_exp)
print(list_result)

# 转换为元祖
gen_exp = (x**2 for x in range(10))
tuple_result = tuple(gen_exp)
print(tuple_result)  # 输出: (0, 1, 4, 9, 16, 25, 36, 49, 64, 81)

# 转换为集合
gen_exp = (x**2 for x in range(10))
set_result = set(gen_exp)
print(set_result)  # 输出: {0, 1, 4, 36, 9, 16, 49, 64, 25, 81}

# 字符串
# 生成器表达式不能直接创建字符串，但可以通过将生成器表达式的结果传递给 ''.join() 函数来创建字符串
# 生成一个包含字符 'a' 到 'j' 的字符串
str_gen = ''.join(chr(97+x) for x in range(10))
print(str_gen)

# 字典
# 使用字典推导式而不是生成器表达式，因为生成器表达式本身不能直接创建字典。但可以通过传递生成器表达式的结果来创建字典：
# 生成一个包含数字及其平方的字典
dict_gen = {x: x**2 for x in range(10)}
print(dict_gen)

# 嵌套循环
# 创建一个包含所有可能的 (x, y) 组合的元组，其中 x 来自于 [0, 1, 2]，y 来自于 [0, 1, 2]：
combinations_gen = ((x,y) for x in range(3) for y in range(3))
print(tuple(combinations_gen))

#  条件和嵌套循环结合
# 创建一个包含所有可能的 (x, y) 组合的元组，但只保留 x 和 y 不相等的组合：
combinations_gen = ((x,y) for x in range(3) for y in range(3) if x != y)
print(tuple(combinations_gen))

# 使用条件
# 生成器表达式可以包含条件，以筛选符合条件的项：
gen_exp = (x**2 for x in range(10) if x% 2==0)
# 使用生成器
for num in gen_exp:
    print(num)

# 结合函数
# 生成器表达式可以与函数一起使用。例如，计算平方和：
gen_exp = (x**2 for x in range(10))
sum_result = sum(gen_exp)
print(sum_result)

