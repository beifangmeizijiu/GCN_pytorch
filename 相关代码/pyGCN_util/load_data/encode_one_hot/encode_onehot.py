"""
encode_onehot 是一个常见的函数，用于将分类标签（categorical labels）转换为独热编码（one-hot encoding）。独热编码是一种表示分类数据的方法，其中每个类别使用一个二进制向量表示，向量中只有一个元素为1，其余元素为0。
下面是一个示例代码，展示了如何使用 encode_onehot 函数将分类标签转换为独热编码：
"""
import numpy as np

def encode_onehot(labels):
    # 获取标签的唯一值
    # unique_labels = np.unique(labels)
    # 或者使用set（）集合也可去重
    unique_labels = set(labels)
    # 创建一个字典，将每个唯一标签映射到一个独热编码向量
    label_to_onehot = {label : np.eye(len(unique_labels))[i] for i, label in  enumerate(unique_labels)}
    # 使用该字典将每个标签转换为对应的独热编码向量
    # onehot_encoded = np.array([label_to_onehot[label] for label in labels])
    # 或者使用get方法获取字典值
    onehot_encoded = np.array(list(map(label_to_onehot.get, labels)), dtype=np.int32)
    return onehot_encoded

# 示例数据
categories = ['cat', 'dog', 'fish', 'cat', 'dog', 'dog', 'cat']

# 将分类标签转换为独热编码
onehot_encoded_categories = encode_onehot(categories)

# 打印结果
print("Original labels:", categories)
print("One-hot encoded labels:\n", onehot_encoded_categories)








