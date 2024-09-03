import scipy.io as scio
import torch
import numpy as np
from sklearn.cluster import KMeans
import torch.nn.functional as Fun
import torch.nn as nn


# 从.mat文件加载数据集，该文件包含多个视图的数据。
def load_data(name, views):
    """
        加载指定.mat文件中的多视图数据集。

        参数:
            name (str): 数据集的名称。
            views (int): 数据集中的视图数量。

        返回:
            X (list of torch.Tensor): 每个视图的数据，转换为torch.Tensor。
            labels (numpy.ndarray): 数据集的标签，格式为一维数组。
    """
    path = 'data/{}.mat'.format(name)
    data = scio.loadmat(path)
    labels = data['Y']
    labels = np.reshape(labels, (labels.shape[0],))

    X = []
    for i in range(0, views):
        tmp = data['X' + str(i + 1)]
        tmp = tmp.astype(np.float32)
        X.append(torch.from_numpy(tmp).to(dtype=torch.float))

    return X, labels


def random_split(X, Y, train_size=0.7):
    """
        将数据随机分割为训练集和测试集。

        参数:
            X (list of torch.Tensor): 多视图数据。
            Y (numpy.ndarray): 标签数组。
            train_size (float): 训练集占总数据的比例。

        返回:
            X_train, X_test (list of torch.Tensor): 训练和测试数据。
            Y_train, Y_test (torch.Tensor): 训练和测试标签。
        """
    Y = torch.tensor(Y)
    number_class = torch.unique(Y)
    index_train = []
    index_test = []
    for i in range(0, number_class.size(0)):
        indices = torch.nonzero(torch.eq(Y, number_class[i])).squeeze()
        random_indices = torch.randperm(len(indices)).tolist()
        indices_train = random_indices[0:int(train_size * len(indices))]
        indices_test = random_indices[int(train_size * len(indices)):]
        index_train.extend(indices[indices_train])
        index_test.extend(indices[indices_test])

    X_train = []
    X_test = []
    for i in range(0, len(X)):
        X_train.append(X[i][index_train, :])
        X_test.append(X[i][index_test, :])

    Y_train = Y[index_train]
    Y_test = Y[index_test]
    return X_train, X_test, Y_train, Y_test


def distance(X, Y, square=True):
    """
    计算两组样本之间的欧几里得距离。

    参数:
        X (torch.Tensor): 样本集合，维度为d*n。
        Y (torch.Tensor): 样本集合，维度为d*m。
        square (bool): 是否返回距离的平方。

    返回:
        torch.Tensor: 距离矩阵，维度为n*m。
    """
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y
    y = y.repeat(n, 1)
    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result


"""
    基于Clustering-with-Adaptive-Neighbors (CAN)方法构建图。
    参数:
        X (torch.Tensor): 数据点集合，维度为d*n。
        num_neighbors (int): 每个节点的邻居数量。
        links (torch.Tensor): 额外的链接（可选）。

    返回:
        weights, raw_weights (torch.Tensor): 图的权重矩阵。
    """


def build_CAN(X, num_neighbors, links=0):
    """
    Solve Problem: Clustering-with-Adaptive-Neighbors(CAN)
    :param X: d * n
    :param num_neighbors:
    :return: Graph
    """
    size = X.shape[1]
    num_neighbors = min(num_neighbors, size - 1)
    distances = distance(X, X)
    distances = torch.max(distances, torch.t(distances))
    sorted_distances, _ = distances.sort(dim=1)
    top_k = sorted_distances[:, num_neighbors]
    top_k = torch.t(top_k.repeat(size, 1)) + 10 ** -10

    sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
    sum_top_k = torch.t(sum_top_k.repeat(size, 1))
    sorted_distances = None
    torch.cuda.empty_cache()
    T = top_k - distances
    distances = None
    torch.cuda.empty_cache()
    weights = torch.div(T, num_neighbors * top_k - sum_top_k)
    T = None
    top_k = None
    sum_top_k = None
    torch.cuda.empty_cache()
    weights = weights.relu().cpu()
    if links != 0:
        links = torch.Tensor(links).to(X.device)
        weights += torch.eye(size).to(X.device)
        weights += links
        weights /= weights.sum(dim=1).reshape([size, 1])
    torch.cuda.empty_cache()
    raw_weights = weights
    weights = (weights + weights.t()) / 2
    raw_weights = raw_weights.to(X.device)
    weights = weights.to(X.device)
    # weights邻接矩阵
    return weights, raw_weights


def contrastive_loss(S, F, Y, temperature=0.1, zita=0.1):
    """
        计算对比损失，用于学习数据表示。

        参数:
            S, F (torch.Tensor): 两组特征表示。
            Y (torch.Tensor): 标签。
            temperature (float): 控制损失计算的温度参数。
            zita (float): 控制损失计算的其他参数。

        返回:
            torch.Tensor: 损失值。
        """
    samples = S.shape[0]

    S = Fun.normalize(S, p=2, dim=1)
    F = Fun.normalize(F, p=2, dim=1)
    s1 = torch.exp(torch.mm(S, F.T) / temperature)
    s2 = torch.exp(torch.mm(F, S.T) / temperature)

    indicator = (Y.unsqueeze(1) != Y.unsqueeze(0)).float().to(S.device)
    W = torch.mul(indicator, 1 - torch.exp(- distance(S.T, F.T) / zita))
    W.fill_diagonal_(1)

    loss = torch.log(torch.diagonal(s1) / torch.sum(torch.mul(W, s1), dim=1)) + \
           torch.log(torch.diagonal(s2) / torch.sum(torch.mul(W, s2), dim=1))

    loss = -torch.sum(loss) / (2 * samples)
    return loss


def graph_normalize(A):
    """
    归一化图的邻接矩阵。

    参数:
        A (torch.Tensor): 邻接矩阵。

    返回:
        torch.Tensor: 归一化的邻接矩阵。
    """
    degree = torch.sum(A, dim=1).pow(-0.5)
    return (A * degree).t() * degree
