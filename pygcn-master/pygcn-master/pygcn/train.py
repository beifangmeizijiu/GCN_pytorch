from __future__ import division
from __future__ import print_function

import codecs
import csv
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.append(r'../')


from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

# Training settings
# 这段代码是一个典型的命令行参数解析器，使用 Python 标准库中的 `argparse` 模块。它允许你从命令行传递参数给你的脚本。让我解释一下这些参数的含义：
parser = argparse.ArgumentParser()
# - `--no-cuda`: 如果提供了这个参数，则 `args.no_cuda` 会被设置为 `True`，表示禁用 CUDA 训练。CUDA 是用于利用 GPU 进行加速计算的技术，如果你不想使用 GPU，可以使用这个参数来禁用它。
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
# - `--fastmode`: 如果提供了这个参数，则 `args.fastmode` 会被设置为 `True`，表示在训练过程中进行快速验证。这通常用于快速验证模型的训练效果，但可能会牺牲一些准确性。
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
# - `--seed`: 设置随机种子，用于在每次运行时产生可重复的随机结果。这对于调试和结果的可重复性非常重要。
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
# - `--epochs`: 设置训练的周期数，即整个数据集被使用多少次。
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
# - `--lr`: 设置初始学习率，即模型在每一步更新参数时使用的学习率大小。
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
# - `--weight_decay`: 设置权重衰减，它是一种正则化技术，用于防止模型过拟合。
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
# - `--hidden`: 设置隐藏层的单元数，即神经网络中隐藏层的神经元数量。
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
# - `--dropout`: 设置 dropout 率，用于在训练过程中随机丢弃神经元，防止过拟合。
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
# 解析命令参数
args = parser.parse_args()
# 检查使用GPU还是CPU,.cuda是动态的添加属性
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print(f"Using device: {device}")
# 确定随机数种子，以便后可以复现实验
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# 加载数据
# adj:邻接矩阵
# features：特征矩阵
# idx_train：训练集
# idx_val：验证集
# idx_test：测试集
adj, NL, features, labels, idx_train, idx_val, idx_test = load_data()

# 模型以及选择
# nfeat 特征矩阵输入维度
# nhid 隐藏层数
# nclass 分类个数
# dropout 正则化率
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)

NL_model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)

# Adam优化器
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
NL_optimizer = optim.Adam(NL_model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# 有显卡，全部改为显卡训练
if args.cuda:
    model.cuda()
    NL_model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    NL = NL.cuda()

def data_write_csv(file_name, datas):
    file_csv = codecs.open(file_name, "w+", encoding='utf-8')
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")




# 普通训练，即即AXW
test_epoch = np.array([])
test_loss_train = np.array([])
def train(epoch):
    global test_epoch, test_loss_train
    # 1、计时器启动：
    t = time.time()
    # 2、训练模式
    model.train()
    # 3、清空梯度
    optimizer.zero_grad()
    # 4、前向传播
    output = model(features, adj)

    # 5、计算训练损失和准确率：
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    # 6、反向传播和优化
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
       # 快速验证模式
        model.eval()
        output = model(features, adj)
    # 7、计算验证损失和准确率：
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    # 将画图元素填入
    test_epoch = np.append(test_epoch, epoch+1)
    test_loss_train = np.append(test_loss_train, loss_val.cpu().detach().numpy())
    # 结果
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

NL_test_epoch = np.array([])
NL_test_loss_train = np.array([])
# NL训练，即LXW
def train_NL(epoch):
    global NL_test_epoch, NL_test_loss_train
    # 1、计时器启动：
    t = time.time()
    # 2、训练模式
    NL_model.train()
    # 3、清空梯度
    NL_optimizer.zero_grad()
    # 4、前向传播
    output = NL_model(features, NL)

    # 5、计算训练损失和准确率：
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    # 6、反向传播和优化
    loss_train.backward()
    NL_optimizer.step()

    if not args.fastmode:
       # 快速验证模式
        NL_model.eval()
        output = NL_model(features, adj)
    # 7、计算验证损失和准确率：
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    # 将画图元素填入
    NL_test_epoch = np.append(NL_test_epoch, epoch+1)
    NL_test_loss_train = np.append(NL_test_loss_train, loss_val.cpu().detach().numpy())

    # 结果
    print('NL_Epoch: {:04d}'.format(epoch+1),
          'NL_loss_train: {:.4f}'.format(loss_train.item()),
          'NL_acc_train: {:.4f}'.format(acc_train.item()),
          'NL_loss_val: {:.4f}'.format(loss_val.item()),
          'NL_acc_val: {:.4f}'.format(acc_val.item()),
          'NL_time: {:.4f}s'.format(time.time() - t))



def test():
    # 整体测试
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

def test_NL():
    NL_model.eval()
    optput = NL_model(features, NL)
    loss_test = F.nll_loss(optput[idx_test], labels[idx_test])
    acc_test = accuracy(optput[idx_test], labels[idx_test])
    print("NL Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("NL Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

t_total = time.time()
for epoch in range(args.epochs):
    train_NL(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
# Testing
test()
test_NL()

# 绘制对比图像
fig, ax = plt.subplots(figsize=(12,8))

ax.plot(test_epoch, test_loss_train, 'r', label='AXW')
# 绘制第二条曲线
plt.plot(test_epoch, NL_test_loss_train, 'b',label='LXW')

# 定义了图例在那个位置，就是线条比例
ax.legend(loc='upper right')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss train')
ax.set_title('Epoch vs. Loss train')
plt.show()