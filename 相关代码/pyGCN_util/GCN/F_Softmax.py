#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
F.log_softmax 是 PyTorch 中的一个函数，用于计算输入张量在指定维度上的 Log-Softmax。Log-Softmax 是 Softmax 函数的对数形式，经常用于多分类问题的神经网络模型的输出层。在多分类问题中，通常会在最后一层使用 Softmax 函数将模型的输出转换为概率分布，然后使用交叉熵损失函数来计算损失。而使用 Log-Softmax 和负对数似然损失（Negative Log-Likelihood Loss，NLLLoss）是一个等价但数值上更稳定的选择。
F.log_softmax 函数的定义
F.log_softmax 函数位于 torch.nn.functional 模块中，主要参数如下：

input: 输入张量。
dim: 应用 Log-Softmax 的维度，通常在分类问题中为类别维度。
使用场景
Log-Softmax 经常用于多分类问题的输出层，与 torch.nn.NLLLoss 结合使用，因为 NLLLoss 期望接收的输入是对数概率。

示例代码
以下是如何在神经网络中使用 F.log_softmax 的示例。
"""
# 示例代码
# 以下是如何在神经网络中使用 F.log_softmax 的示例。
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # 使用 F.log_softmax

# 示例模型实例
model = SimpleNN(nfeat=784, nhid=128, nclass=10, dropout=0.5)

# 打印模型结构
print(model)

# 创建一个示例输入
input_tensor = torch.randn(1, 784)

# 前向传播
output = model(input_tensor)
print(output)

# 为什么使用 Log-Softmax
# 数值稳定性：在计算交叉熵损失时，使用 Log-Softmax 比先使用 Softmax 再取对数数值上更稳定。直接对 Softmax 的输出取对数可能会导致数值下溢，而 Log-Softmax 将这些操作合并为一个步骤，避免了这种风险。
# 与 NLLLoss 结合使用：torch.nn.NLLLoss 函数期望接收对数概率（log-probabilities），因此与 Log-Softmax 输出自然结合。

# 综合示例
# 以下是一个完整的例子，展示了如何定义和使用包含 Log-Softmax 的 GCN 模型：

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / self.weight.size(1)**0.5
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_features} -> {self.out_features})'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def __repr__(self):
        return (f'{self.__class__.__name__}(\n'
                f'  (gc1): {self.gc1}\n'
                f'  (relu): ReLU()\n'
                f'  (dropout): Dropout(p={self.dropout})\n'
                f'  (gc2): {self.gc2}\n'
                f'  (log_softmax): LogSoftmax(dim=1)\n'
                f')')

# 示例模型实例
model = GCN(nfeat=1433, nhid=16, nclass=7, dropout=0.5)
print(model)

# 创建示例输入
input_tensor = torch.randn(5, 1433)  # 5 个节点，每个节点有 1433 维特征
adj_matrix = torch.eye(5)  # 简单的 5x5 单位邻接矩阵

# 前向传播
output = model(input_tensor, adj_matrix)
print(output)



