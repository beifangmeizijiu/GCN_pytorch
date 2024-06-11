"""
numpy.genfromtxt 是一个用于从文本文件加载数据的函数，它能够处理缺失数据并进行类型转换。这个函数非常适合用来读取包含数值数据的文件（如 CSV 文件），即使文件中包含一些缺失值。
"""
import numpy as np
# 参数解释
# fname: 文件名或生成器（可以是字符串、路径对象或包含数据的文件对象）。
# dtype: 数据类型，可选。默认是 float。
# delimiter: 分隔符。默认为空格。
# skip_header: 跳过文件开始的行数。
# skip_footer: 跳过文件结尾的行数。
# missing_values: 要被视为缺失值的字符串或字符串序列。
# filling_values: 用于填充缺失值的值。
# usecols: 指定要读取的列。可以是整数、元组或列表。
# names: 如果 True，则将第一行作为列名读取；如果为列表，则将其用作列名。
# encoding: 文件的编码。默认是 None。

# 基本用法
data = np.genfromtxt('./data.csv', delimiter=',')
print(type(data),'\n',data)

# 处理缺失值
data = np.genfromtxt('./data_1.csv', delimiter=',', filling_values=0)
print("缺失值填充0",data)

# 读取指定列
data = np.genfromtxt('./data.csv', delimiter=',',usecols=(1,))
print("只读取第二列",data)

# 使用列名
data = np.genfromtxt('./data_2.csv', delimiter=',',names=True)
print("使用列名读取",data)
print(data['B'])



