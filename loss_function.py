import torch
import torch.nn.functional as F


# 计算 Kullback-Leibler (KL) 散度，用于衡量两个 Dirichlet 分布之间的差异：
def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


# 计算对数似然损失，用于衡量预测和真实标签之间的差异：
def loglikelihood_loss(y, alpha, device):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


# 计算均方误差（MSE）损失，并在需要时添加 KL 散度损失：
def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None, useKL=True):
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    if not useKL:
        return loglikelihood

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


# 计算基于函数（如对数函数或 Digamma 函数）的 EDL 损失，并在需要时添加 KL 散度损失：
def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device, useKL=True):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    if not useKL:
        return A

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


# 分别计算基于 MSE、对数函数和 Digamma 函数的 EDL 损失：
def edl_mse_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    return torch.mean(loss)


def edl_log_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = edl_loss(torch.log, target, alpha, epoch_num, num_classes, annealing_step, device)
    return torch.mean(loss)


def edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device)
    return torch.mean(loss)


# 定义多视图证据一致性损失，基于MRF的参数edge_weight计算冲突度
def get_mrf_loss(evidences, regularization_strength=0.01, l2_strength=0.0001, similarities_strength=1):
    num_views = len(evidences)
    similarities = []
    all_evidences = torch.stack(list(evidences.values()))  # 将所有视图的证据堆叠成一个新的张量

    # 计算每对证据之间的相似度
    for i in range(num_views):
        for j in range(i + 1, num_views):
            similarity = F.cosine_similarity(all_evidences[i], all_evidences[j], dim=1).mean()
            similarities.append(similarity)

    similarities = torch.tensor(similarities)
    mean_similarity = similarities.mean()  # 计算所有相似度的平均值
    # 计算相似度损失
    similarity_loss = - mean_similarity * similarities_strength
    # 计算多样性损失：每个视图证据的方差
    diversity_loss = torch.var(all_evidences, dim=1).mean()  # 计算所有视图的方差并求均值
    # 计算L2正则项
    l2_reg = 0
    for i in range(num_views):
        l2_reg += torch.norm(all_evidences[i], p=2)  # 计算每个视图证据的L2范数
    l2_reg *= l2_strength  # 乘以正则化强度
    # 结合相似度损失、多样性损失和L2正则项
    total_loss = similarity_loss + regularization_strength * diversity_loss - l2_reg
    return total_loss


# 计算总损失，包括分类损失和一致性损失
def get_loss(evidences, evidence_a, target, epoch_num, num_classes, annealing_step, gamma, device):
    target = F.one_hot(target, num_classes).float()
    alpha_a = evidence_a + 1
    loss_acc = edl_digamma_loss(alpha_a, target, epoch_num, num_classes, annealing_step, device)
    for v in range(len(evidences)):
        alpha = evidences[v] + 1
        loss_acc += edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device)
    loss_acc = loss_acc / (len(evidences) + 1)
    # 加入基于MRF边权重的一致性损失
    mrf_loss = get_mrf_loss(evidences)
    total_loss = loss_acc + gamma * mrf_loss
    return total_loss
