"""
preds.eq 是 PyTorch 中用于元素级比较的一个方法，它返回一个布尔张量，指示两个张量对应位置的元素是否相等。在分类任务中，它常用于比较模型预测的类别和真实的类别，以计算准确率。
用法
preds.eq 的典型用法如下：
result = preds.eq(other)
preds 是一个张量，通常是模型的预测结果。
other 是另一个张量，通常是实际的目标标签。
result 是一个布尔张量，表示 preds 和 other 对应位置的元素是否相等。
"""
# 示例
# 以下是一个完整的例子，演示如何使用 preds.eq 计算预测的准确率：
import torch
# 模拟批量大小为5的模型预测类别
preds = torch.tensor([0, 2, 1, 3, 1])

# 实际目标标签
labels = torch.tensor([0, 1, 1, 3, 0])

# 使用 eq 方法比较 preds 和 labels

correct = preds.eq(labels)

# 打印结果
print("Comparison result (correct):", correct)  # 输出：tensor([ True, False,  True,  True, False])

# 计算准确率
accuracy = correct.sum().item() / len(labels)
print("Accuracy:", accuracy)  # 输出：0.6






