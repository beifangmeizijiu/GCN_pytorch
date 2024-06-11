"""
F.nll_loss 是 PyTorch 中的一个损失函数，用于计算负对数似然损失（Negative Log Likelihood Loss）。它通常用于多分类任务中，尤其是在使用 log_softmax 激活函数的输出之后。
公式
假设
x 是模型的输出（经过 log_softmax 处理后的对数概率），
y 是目标标签，负对数似然损失的公式为：
Loss(x,y)=− 1/N sum(i,n)xi,yi
 sum(i,n)计算xi,yi从i到n求和
 其中 𝑥𝑖,𝑦𝑖 表示第 𝑖个样本中目标类别 𝑦𝑖 的对数概率，N 是样本数量。

 F.nll_loss（负对数似然损失）函数和目标标签之间的关系可以通过理解其工作机制来说明。简而言之，F.nll_loss 使用目标标签来选择模型输出中的相应对数概率，并计算这些对数概率的负值的平均值或总和作为损失。

工作机制
输入和目标标签：

输入：模型输出经过 log_softmax 处理后的对数概率（log probabilities），通常形状为 (N,C)，其中
N 是批量大小，
C 是类别数。
目标标签：一个包含目标类别索引的张量，形状为 (N)，其中每个值是一个从 0 到 C−1 之间的整数，表示每个样本的正确类别。

损失计算：
F.nll_loss 会根据目标标签，从输入张量中选取对应类别的对数概率。
对这些选取的对数概率取负值。
根据 reduction 参数（默认为 'mean'），计算这些负对数概率的平均值或总和。

函数定义
torch.nn.functional.nll_loss(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
input: 经过 log_softmax 处理后的张量，形状为 (N,C) 或(𝑁,𝐶,𝑑1,𝑑2,...，dk),N 是批量大小，C 是类别数。
target: 目标标签，形状为(N) 或(𝑁,𝐶,𝑑1,𝑑2,...，dk),)，包含每个样本的正确类别的索引。
weight: 一个可选的手动重新加权类别的权重张量，形状为(C)。
size_average: 已被弃用。若设为 True，平均损失；否则总损失和。使用 reduction 参数代替。
ignore_index: 可选的整数，指定一个类别索引，在计算损失时将忽略该类别。
reduce: 已被弃用。若设为 True，返回总损失；否则返回未聚合的损失。使用 reduction 参数代替。
reduction: 指定应用于输出的方式，取值包括 'none' | 'mean' | 'sum'。默认值为 'mean'。
"""


"""
通过 log_softmax 转换为对数概率后，值越大，对应类别的概率越高。这是因为 log_softmax 是 softmax 函数的对数形式，它将输入值（logits）转换为对数概率。

详细解释
Softmax 函数：

Softmax 函数将原始的 logits 转换为概率分布。具体公式如下：
softmax(𝑥𝑖)=𝑒^𝑥𝑖 / sum(j,n)e^xj
其中 xi是原始的 logit 值。

Log Softmax 函数：

Log Softmax 是 Softmax 函数的对数形式，它将 logits 转换为对数概率。具体公式如下：
log_softmax(𝑥𝑖)=log(softmax(𝑥𝑖))=𝑥𝑖−log(sum(j,n)e^xj)
其中 xi是原始的 logit 值。
由于 Log Softmax 是对 Softmax 的对数操作，所以 log_softmax 的输出值越大，表示该类别的对数概率越高，相应的原始概率也越高。因此，值越大的对数概率表明模型认为该类别的可能性越高。
"""

# 示例代码

import torch
import torch.nn.functional as F

# 模拟批量大小为3，类别数为5的输出 (logits)
logits = torch.tensor([[1.2, 0.5, -0.8, 2.4, 1.3],
                       [0.3, 1.8, 0.9, -0.5, -1.2],
                       [-1.1, 2.2, 0.1, 0.9, 1.4]])

# 目标标签
target = torch.tensor([3, 1, 4])

# 计算log_softmax
log_prob = F.log_softmax(logits, dim=1)
# 打印 log_softmax 的结果
print("Log probabilities:\n", log_prob)

# 计算 softmax 以得到概率
prob = torch.exp(log_prob)

# 打印 softmax 的结果
print("Probabilities:\n", prob)

# 在这个例子中，logits 经过 log_softmax 转换为对数概率 log_prob。可以看到，log_prob 中值最大的元素对应的类别，在 prob 中具有最高的概率。例如，第一个样本中，对数概率最大的值是 -0.7117（对应原始的 2.4），它对应的概率是 0.4909，这是所有类别中最大的概率

# 使用 F.nll_loss 计算
loss = F.nll_loss(log_prob, target)
print(loss)

# 计算步骤
# 选择目标类别的对数概率：
#
# 对于第一个样本，目标类别是 3，对应的对数概率是 -0.6012。
# 对于第二个样本，目标类别是 1，对应的对数概率是 -0.5765
# 对于第三个样本，目标类别是 4，对应的对数概率是 -1.4319。
# 取负值并计算平均值：
#
# 负对数似然损失为：-(-0.6012) + -(-0.5765) + -(-1.4319) = 2.6096
# 即 0.6012 + 0.5765 + 1.4319 = 2.6096

# 如果使用默认的 reduction='mean'，则损失为 2.6096 / 3 = 0.8699
