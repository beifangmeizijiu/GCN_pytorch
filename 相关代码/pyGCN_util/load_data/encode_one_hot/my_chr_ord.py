"""
chr 和 ord 是 Python 中的两个内置函数，分别用于字符与其对应的 Unicode 码点之间的相互转换。

chr 函数
作用：将 Unicode 码点（整数）转换为对应的字符。
语法：chr(i)
i：整数，表示 Unicode 码点。有效范围是 0 到 1,114,111（即 0x10FFFF）。
返回值：一个字符串，表示 Unicode 字符 i 对应的字符。
"""
# 示例
print(chr(65))      # 输出: 'A'
print(chr(97))      # 输出: 'a'
print(chr(8364))    # 输出: '€'（欧元符号）
print(chr(128512))  # 输出: '😀'（笑脸表情符）

"""
ord 函数
作用：将单个字符转换为其对应的 Unicode 码点（整数）。
语法：ord(c)
c：字符串，长度为1。
返回值：一个整数，表示字符 c 对应的 Unicode 码点。
"""
# 示例

print(ord('A'))     # 输出: 65
print(ord('a'))     # 输出: 97
print(ord('€'))     # 输出: 8364
print(ord('😀'))    # 输出: 128512

"""
chr 和 ord 的常见用法
处理 ASCII 字符：适用于 ASCII 字符集（码点 0 到 127）中的字符。
处理 Unicode 字符：适用于全范围 Unicode 字符，包括各种语言字符、符号和表情符号。
字符编码转换：可以在字符和 Unicode 码点之间自由转换，有助于字符编码和解码。
"""





