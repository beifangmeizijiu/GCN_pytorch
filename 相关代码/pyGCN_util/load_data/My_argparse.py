"""
argparse的使用
argparse 是 Python 标准库中的一个模块，用于轻松编写用户友好的命令行接口。

"""
# 假设你要编写一个Python脚本，该脚本接受三个参数：一个必需的文件名，一个可选的整数参数，以及一个可选的布尔标志。以下是如何实现的示例
import argparse

# 创建一个解析器对象；创建一个ArgumentParser对象，并为脚本添加描述。
parser = argparse.ArgumentParser(description="这是一个简单的示例脚本")

# 添加必需的参数；
parser.add_argument('filename', type=str, help='输入文件的名称')

# 添加可选的整数参数;并自定义成员变量名
parser.add_argument('--number', type=int, default=10, help='一个可选的整数参数，默认值为10', dest='number_value')

# 添加可选的布尔标志
parser.add_argument('--verbose', action='store_true', help='是否输出详细信息')

# 解析命令行参数
args = parser.parse_args()

# 使用解析后的参数
print(f"文件名: {args.filename}")
print(f"整数参数: {args.number_value}")
print(f"详细模式: {'开启' if args.verbose else '关闭'}")

# 示例功能
if args.verbose:
    print("详细模式已开启，开始执行更多的调试信息...")
# 可以在这里添加你的脚本功能

# 使用python pymy_argparse.py input.txt --number 42 --verbose
