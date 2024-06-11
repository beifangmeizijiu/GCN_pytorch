import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        # 构建特征向量和ID图框架
        self.gc1 = GraphConvolution(nfeat, nhid)
        # 构建为ID图和分类图框架
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        # 使用ReLU激活函数f(x)=max(0,x)
        # 1、先使用线性模型向前传播
        # 2、使用激活函数
        x = F.relu(self.gc1(x, adj))
        # 正则化项防止过拟合
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        #  Softmax 分类函数
        return F.log_softmax(x, dim=1)
