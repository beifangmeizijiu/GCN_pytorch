import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

# 图卷积神经网络
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    # in_features输入特征矩阵的输入个数
    # out_features输出特征矩阵的输出个数
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 权重参数初始化Kaiming方法
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 前向传播
    def forward(self, input, adj):
        # 输入向量 * 权重
        support = torch.mm(input, self.weight)
        # 对称邻接矩阵 * 输入向量 * 权重
        output = torch.spmm(adj, support)
        #  输出+偏执值
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    # 打印每一层的定义
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
