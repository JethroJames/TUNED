import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个图卷积层GCNLayer，它继承自nn.Module
class GCNLayer(nn.Module):
    # 初始化函数，参数为输入特征数和输出特征数
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()  # 调用父类构造函数
        self.linear = nn.Linear(in_features, out_features)  # 定义一个线性层

    # 前向传播函数，参数为特征矩阵X和邻接矩阵A
    def forward(self, X, A):
        A_hat = A + torch.eye(A.size(0)).to(A.device)  # 计算正则化的邻接矩阵
        D_hat = torch.diag(torch.sum(A_hat, dim=1))  # 计算度矩阵
        D_hat_inv_sqrt = torch.inverse(torch.sqrt(D_hat))  # 计算度矩阵的逆平方根
        A_norm = torch.mm(torch.mm(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)  # 计算规范化的邻接矩阵
        out = torch.mm(A_norm, X)  # 图卷积操作
        out = self.linear(out)  # 应用线性变换
        return out


# 定义整个GCN网络，也是继承自nn.Module
class GCN(nn.Module):
    def __init__(self, out_features, class_num):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(out_features, out_features)  # 第一层GCN
        self.gcn2 = GCNLayer(out_features, class_num)  # 第二层GCN，输出分类数

    # todo:X = F.relu(self.gcn2(self.gcn1(X, A), A))  # 通过两层GCN并应用ReLU激活函数;softplus() relu() exp()
    def forward(self, X, A):
        X = F.softplus(self.gcn2(self.gcn1(X, A), A))
        return X


# 定义证据收集器，每个视图用来处理输入并预测类别
class EvidenceCollector(nn.Module):
    def __init__(self, dims, num_classes):
        super(EvidenceCollector, self).__init__()
        self.num_layers = len(dims)  # 层数
        self.net = nn.ModuleList()  # 网络层列表
        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))  # 添加线性层
            self.net.append(nn.ReLU())  # 添加ReLU激活层
            self.net.append(nn.Dropout(0.1))
        self.net.append(nn.Linear(dims[self.num_layers - 1], num_classes))
        self.net.append(nn.Softplus())  # 使用Softplus作为最后的激活函数

    def forward(self, x):
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        return h


# 定义一个多视图分类器RCML，用于收集来自多个视图的证据，并进行融合
class TUNED(nn.Module):
    def __init__(self, num_views, dims, num_classes, psi=0.7):
        super(TUNED, self).__init__()
        self.num_views = num_views  # 视图数量
        self.num_classes = num_classes  # 类别数
        self.psi = psi  # Threshold factor for edge inclusion
        # 创建一个模块列表，每个视图对应一个EvidenceCollector
        self.EvidenceCollectors = nn.ModuleList(
            [EvidenceCollector(dims[i], self.num_classes) for i in range(self.num_views)])
        # 创建一个模块列表，每个视图对应一个GCN
        self.GCNs = nn.ModuleList([
            GCN(dims[i][0], num_classes)  # Adjust GCN input and output dimensions here
            for i in range(self.num_views)
        ])
        # 融合
        # self.weights = nn.ParameterList([
        #     nn.Parameter(torch.rand(2)) for _ in range(num_views)
        # ])

        # Consensus Evidence Generator
        self.alpha = nn.Parameter(torch.ones(num_classes))  # 用于迪利克雷分布的参数

        # 添加MultiheadAttention模块
        self.attention = nn.MultiheadAttention(embed_dim=num_classes, num_heads=1)

    def forward(self, X, A, ce_scale=0.1, consensus_evidence_type='dirichlet'):
        # 收集每个视图的证据
        evidences = dict()
        for v in range(self.num_views):
            # 结构嵌入和特征嵌入特征融合(相加)
            raw_evidence = self.EvidenceCollectors[v](X[v]) + self.GCNs[v](X[v], A[v])
            # 为每个视图的证据添加协同证据
            if consensus_evidence_type == 'dirichlet':
                consensus_evidence = ce_scale * (
                    torch.distributions.Dirichlet(self.alpha).sample((raw_evidence.size(0),)).to(
                        raw_evidence.device))
            elif consensus_evidence_type == 'uniform':
                consensus_evidence = ce_scale * (torch.rand_like(raw_evidence))
            elif consensus_evidence_type == 'gaussian':
                consensus_evidence = ce_scale * (torch.randn_like(raw_evidence) * self.sigma + self.mu)
            else:
                raise ValueError("Unsupported noise type")
            evidences[v] = consensus_evidence+raw_evidence

        # 结合MRF融合不同视图的证据
        edge_index, edge_weights = self.create_complete_graph(self.num_views, evidences)
        evidence_a = self.mrf_aggregate(evidences, edge_index, edge_weights)
        return evidences, evidence_a

    def create_complete_graph(self, num_nodes, evidences):
        edges = []
        edge_weights = []
        all_similarities = []

        # 首先计算所有视图间的相似度
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                similarity = F.cosine_similarity(evidences[i], evidences[j], dim=1)
                all_similarities.append(similarity.mean().item())
        similarities_tensor = torch.tensor(all_similarities)
        normalized_similarities = similarities_tensor
        max_similarity = max(normalized_similarities)  # 找到最大相似度
        threshold = self.psi * max_similarity  # 设置阈值为最大相似度的psi

        # 再次循环以确定哪些边应该被包括
        index = 0
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                similarity = normalized_similarities[index]
                index += 1
                if similarity > threshold:
                    edges.append((i, j))
                    edge_weights.append(similarity.item())  # 添加边权重
                else:
                    edges.append((i, j))
                    edge_weights.append(0)  # 不符合条件的边权重设置为0

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        return edge_index, edge_weight

    def mrf_aggregate(self, evidences, edge_index, edge_weight):
        # 初始化融合证据为零向量
        aggregated_evidence = torch.zeros_like(next(iter(evidences.values())))
        # 聚合算法
        # print(edge_index[0],edge_index[1])
        for i, evidence in evidences.items():
            # 找到与视图 i 相连接的所有视图
            connections = (edge_index[0] == i).nonzero(as_tuple=True)[0]
            # 对与视图 i 相连接的边权重进行 softmax 归一化
            connected_weights = edge_weight[connections]
            normalized_weights = F.softmax(connected_weights, dim=0)
            # 对每个连接的视图 j 进行证据聚合
            for idx, weight in zip(connections, normalized_weights):
                j = edge_index[1][idx].item()  # 转换为整数
                if j in evidences:  # 确保引用有效
                    aggregated_evidence += weight * evidences[j]
                    aggregated_evidence += weight * evidences[i]
                else:
                    print(f"Warning: Missing evidence for view {j}")

        # 归一化融合证据，确保保持总量不变
        num_connections = len(evidences)
        if num_connections > 0:
            aggregated_evidence /= num_connections * 2
        return aggregated_evidence