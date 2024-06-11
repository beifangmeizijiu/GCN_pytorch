import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


def encode_onehot(labels):
    # 获取唯一标签
    classes = set(labels)
    # 创建一个字典，将每个唯一标签映射到一个独热编码向量
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    # 使用该字典将每个标签转换为对应的独热编码
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

# 数据加载
def load_data(path="../data/cora/", dataset="cora"):
    """加载引文网络的数据集 (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    # 加载数据
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))

    # 建立特征矩阵(使用行稀疏矩阵)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

    # 将分类标签（categorical labels）转换为独热编码（one-hot encoding）
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    # 论文id字典
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    # 生成边
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)

    # 建立邻接矩阵
    # 1、将被引用论文id-引用论文id转为被引用论文编号-引用论文编号
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    # 2、建立邻接矩阵
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # 建立对称邻接矩阵
    # 具体操作步骤：
    # 1、保留adj的转置存在而adj中不存在的项
    # 2、去除adj中存在，同时adj的转置中也存在的项
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 特征矩阵归一化处理
    features = normalize(features)

    # NL = normalized graph Laplacian
    NL = Graph_Laplacian(adj)

    # 邻接矩阵+单位阵（结合自身节点的信息）初学注意区分np.eye和sp.eye
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = normalize((adj + sp.eye(adj.shape[0])))
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    # 特征矩阵从稀疏矩阵依次转为数组——》浮点型张量
    features = torch.FloatTensor(np.array(features.todense()))
    # 获取不同类型的行编码
    labels = torch.LongTensor(np.where(labels)[1])
    # 转换为稀疏张量
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    NL = sparse_mx_to_torch_sparse_tensor(NL)
    # 划分为训练集、验证集、测试集
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)


    return adj, NL, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """行归一化稀疏矩阵"""
    #
    # 具体如下：
    # 1、特征数量n（就是矩阵多少行）
    rowsum = np.array(mx.sum(1))
    # 2、得出1/n
    r_inv = np.power(rowsum, -1).flatten()
    # 3.处理无穷大值为0
    r_inv[np.isinf(r_inv)] = 0.
    # 构造对角矩阵
    r_mat_inv = sp.diags(r_inv)
    # 归一化（ 将对角矩阵与原矩阵相乘，完成行归一化。）
    mx = r_mat_inv.dot(mx)
    return mx

# L = IN − (power(D,−1/2) * A * power(D,−1/2))
# 输入为邻接矩阵
def Graph_Laplacian(adj):
    """将邻接矩阵转入谱图卷积中的拉普拉斯矩阵"""
    # 1、创建单位矩阵（用于自连接）
    IN = np.eye(adj.shape[0])
    # print(IN)
    # 2、创建度矩阵
    D = diags(np.array(adj.sum(axis=1)).flatten(), dtype=float)

    # 3、计算power(D,−1/2)

    """方法一：只适合对角矩阵"""
    # 提取对角元素
    # D_diag = D.diagonal()
    #
    # # 计算对角元素的-1/2次幂，并处理零或者负值
    # D_diag_inv_sqrt = np.where(D_diag > 0, D_diag ** (-0.5), 0)
    #
    # # 处理无穷大值为0
    # D_diag_inv_sqrt[np.isinf(D_diag_inv_sqrt)] = 0.
    #
    # # 构建新的稀疏对角矩阵
    # sqrt_inv_D_matrix = diags(D_diag_inv_sqrt, 0)
    """方法二：只适合对角矩阵"""
    # 1、特征数量n（就是矩阵多少行）
    rowsum = np.array(D.sum(1))
    # 2、得出-0.5次方
    r_inv = np.power(rowsum, -0.5).flatten()
    # 3.处理无穷大值为0
    r_inv[np.isinf(r_inv)] = 0.
    # 构造对角矩阵
    sqrt_inv_D_matrix = sp.diags(r_inv)

    tmp = sqrt_inv_D_matrix @ adj @ sqrt_inv_D_matrix
    # tmp = D - adj
    tmp = tmp.toarray()

    # 4、计算L = IN − (power(D,−1/2) * A * power(D,−1/2))
    return sp.csr_matrix(IN - tmp)

# 计算精确度
def accuracy(output, labels):
    # 取最大可能性的类别；将预测的类别张量转换为和标签张量相同的数据类型。
    preds = output.max(1)[1].type_as(labels)
    # 对比预测是否正确
    correct = preds.eq(labels).double()
    # 算预测正确总和
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将稀疏矩阵转换为Tensor变量."""
    # 1、将矩阵转换为float类型的coo稀疏矩阵
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # 2、得到矩阵的非0行和列的张量
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # 3、获取非0值
    values = torch.from_numpy(sparse_mx.data)
    # 4、获取矩阵大小
    shape = torch.Size(sparse_mx.shape)
    # 创建稀疏张量
    return torch.sparse.FloatTensor(indices, values, shape)

if __name__ == '__main__':
    # 加载数据
    adj, NL,features, labels, idx_train, idx_val, idx_test = load_data()
