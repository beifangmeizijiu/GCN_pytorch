"""
在 PyTorch 中，优化器用于更新模型的参数以最小化损失函数。optim.Adam 是一种常用的优化器，它基于自适应矩估计（Adaptive Moment Estimation）。Adam 优化器结合了 AdaGrad 和 RMSProp 的优点，能够在训练过程中动态调整学习率，从而加速收敛。
optim.Adam 函数
optim.Adam 的主要参数如下：

params: 待优化的参数。这通常是通过 model.parameters() 获取的模型参数。
lr: 学习率，控制每次参数更新的步长。默认值为 0.001。
weight_decay: 权重衰减（L2 正则化系数），用于防止过拟合。默认值为 0。
"""
# GCN 示例
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import  Dataset, DataLoader



class GraphConvolution(nn.Module):
    def __init__(self,in_features, out_features, bias=True):
        super(GraphConvolution,self).__init__()
        self.in_features=in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features,out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / self.weight.size(1) ** 0.5
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


# 定义示例模型实例
model = GCN(nfeat=1433, nhid=16, nclass=7, dropout=0.5)
# 假设 args 包含 lr 和 weight_decay 参数
class Args:
    def __init__(self, lr, weight_decay):
        self.lr = lr
        self.weight_decay = weight_decay

args = Args(lr=0.01, weight_decay=5e-4)

# 创建一个自定义数据集
# class GraphDataset(Dataset):
#     def __init__(self):
#         # 示例数据：这里我们假设有 5 个节点，每个节点有 1433 维特征
#         self.data = [(torch.rand(5, 1433), torch.eye(5))]
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx]

# 初始化 Adam 优化器
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# 假设 dataloader 提供训练数据
# 示例数据加载器和数据
# dataset = GraphDataset()
# dataloader = DataLoader(dataset, batch_size=1)

dataloader = [(torch.rand(5, 1433), torch.eye(5))]
# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    for data in dataloader:
        x, adj = data  # 假设 data 包含特征矩阵和邻接矩阵
        optimizer.zero_grad()  # 清空梯度
        output = model(x, adj)   # 前向传播
        target = torch.tensor([0, 1, 2, 3, 4])  # 示例目标张量
        loss = F.nll_loss(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

    print(f'Epoch {epoch}, Loss: {loss.item()}')