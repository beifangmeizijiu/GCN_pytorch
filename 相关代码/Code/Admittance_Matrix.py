import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 创建一个简单的无向图
G = nx.Graph()
edges = [(0, 1), (0, 2), (1, 2), (1, 3)]
G.add_edges_from(edges)

# 计算邻接矩阵
A = nx.adjacency_matrix(G).todense()
print("邻接矩阵 A:")
print(A)

# 计算度矩阵
D = np.diag([d for n, d in G.degree()])
print("度矩阵 D:")
print(D)

# 计算拉普拉斯矩阵
L = D - A
print("拉普拉斯矩阵 L:")
print(L)

# 计算拉普拉斯矩阵的特征值和特征向量
eigvals, eigvecs = np.linalg.eigh(L)
print("拉普拉斯矩阵的特征值:")
print(eigvals)
print("拉普拉斯矩阵的特征向量:")
print(eigvecs)

# 绘制图，并显式地设置 random_state 参数以确保可重复性
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
plt.title('Graph Visualization')
plt.show()
