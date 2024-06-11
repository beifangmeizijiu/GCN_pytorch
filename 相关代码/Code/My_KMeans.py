import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
# 设置环境变量以避免内存泄漏问题
os.environ['OMP_NUM_THREADS'] = '1'

# 创建一个简单的无向图
G = nx.Graph()
edges = [(0, 1), (0, 2), (1, 2), (1, 3)]
G.add_edges_from(edges)

# 计算邻接矩阵
A = nx.adjacency_matrix(G).todense()
print("邻接矩阵 A:")
print(A)

# 计算度矩阵
D = np.diag([d for n , d in G.degree()])
print("度矩阵 D:")
print(D)

# 计算拉普拉斯矩阵
L = D - A
print("拉普拉斯矩阵 L:")
print(L)

# 计算拉普拉斯矩阵的特征值和特征向量
eigvals, eigvecs = np.linalg.eigh(L)
print("拉普拉斯矩阵的特征值")
print(eigvals)
print("拉普拉斯矩阵的特征向量：")
print(eigvecs)

# 绘制原始图，并显式地设置 random_state 参数以确保可重复性
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color = 'lightblue', edge_color = 'red')
plt.title('Graph Visualization')
plt.show()

# 利用谱聚类对图进行划分
from sklearn.cluster import KMeans

# 选择特征向量进行聚类
k = 2 # 假设我们将图分成两个簇
eigvecs_k = eigvecs[:,:k]

# 使用KMeans进行聚类，并显式设置 n_init 参数以避免 FutureWarning
# .fit()将数据传入模型
kmeans = KMeans(n_clusters = k, n_init=10, random_state = 42).fit(eigvecs_k)

"""
1. 聚类标签（Cluster Labels）
聚类标签表示每个数据点所属的簇（cluster）。在 KMeans 算法中，算法会将数据点分配到预定数量的簇中，每个簇由一个标签表示。标签通常是从 0 开始的整数，表示每个数据点属于哪个簇。

例如，如果有两个簇，标签可能是 0 或 1，表示数据点属于第一个簇或第二个簇。

2. 聚类中心（Cluster Centers）
聚类中心是每个簇的中心点，通常是该簇中所有数据点的均值。它表示该簇的平均位置。KMeans 算法通过迭代调整簇中心的位置，直到簇内数据点之间的距离最小化，从而找到最佳的簇划分。
"""
# 获取聚类标签
labels = kmeans.labels_
print("聚类标签:", labels)

# 获取聚类中心
centers = kmeans.cluster_centers_
print("聚类中心:", centers)
# 根据聚类结果绘图
colors = ['lightblue' if label == 0 else 'lightgreen' for label in labels]
nx.draw(G, pos, with_labels=True, node_color = colors, edge_color = 'red')

plt.title('Graph Clustering')
plt.show()

# 绘制数据点
plt.scatter(eigvecs_k[:, 0], eigvecs_k[:, 1], c=labels, cmap='viridis')

# 绘制聚类中心
plt.scatter(centers[:, 0], centers[:, 1], s=300, c='red', marker='X')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('KMeans Clustering')
plt.show()


