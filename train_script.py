# 导入必要的库
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import importlib
import argparse
from loss_function import get_loss
import utils  # 导入构建图结构的函数
# 设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 设置随机种子
seed = 12  # 你可以选择任何你喜欢的种子值
set_seed(seed)

# 设置NumPy的打印选项，以优化数字的显示
np.set_printoptions(precision=4, suppress=True)


# 定义构建图的函数，用于生成图的邻接矩阵
def build_graphs(X, neighbor):
    A = []
    # 遍历每个视图的数据
    for v in range(len(X)):
        X_v = torch.tensor(X[v]).T  # 转置数据
        A_v, _ = utils.build_CAN(X_v, neighbor)  # 使用工具函数构建邻接矩阵
        A.append(A_v)  # 添加到列表
    return A


# 定义应对冲突的训练流程（略）
def train_and_evaluate(args):
    # 动态导入数据集和模型
    data_module = importlib.import_module('data')
    model_module = importlib.import_module(f'model.{args.model_path}')
    DatasetClass = getattr(data_module, args.dataset)
    ModelClass = getattr(model_module, 'TUNED')

    dataset = DatasetClass()
    num_samples = len(dataset)
    num_classes = dataset.num_classes
    num_views = dataset.num_views
    dims = dataset.dims
    index = np.arange(num_samples)
    np.random.shuffle(index)
    train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]

    train_loader = DataLoader(Subset(dataset, train_index), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_index), batch_size=args.batch_size, shuffle=False)

    if args.add_conflict:
        dataset.postprocessing(test_index, addNoise=False, sigma=0.5, ratio_noise=0.1, addConflict=True,
                               ratio_conflict=0.4)

    model = ModelClass(num_views, dims, num_classes, psi=args.psi)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gamma = args.gamma

    model.to(device)
    model.train()
    loss_history = []  # 记录每个 epoch 的损失值
    for epoch in range(1, args.epochs + 1):
        consensus_evidence_scale = min(1.0, epoch / args.epochs)  # 例如，线性增长到1
        print(f'====> {epoch}')
        epoch_loss = 0.0
        num_batches = 0
        for X, Y, indexes in train_loader:
            A_train = build_graphs(X, neighbor=args.neighbor)
            for v in range(num_views):
                X[v] = X[v].to(device)
                A_train[v] = A_train[v].to(device)
            Y = Y.to(device)
            evidences, evidence_a = model(X, A_train, ce_scale=consensus_evidence_scale)
            loss = get_loss(evidences, evidence_a, Y, epoch, num_classes, args.annealing_step, gamma, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        average_loss = epoch_loss / num_batches
        loss_history.append(average_loss)  # 记录当前 epoch 的平均损失
        print(f'Average Loss for Epoch {epoch}: {average_loss}')  # 打印当前周期的平均损失

    # 绘制损失函数图
    plt.figure()
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()

    model.eval()
    num_correct, num_sample = 0, 0
    for X, Y, indexes in test_loader:
        A_test = build_graphs(X, neighbor=args.neighbor)
        for v in range(num_views):
            X[v] = X[v].to(device)
            A_test[v] = A_test[v].to(device)
        Y = Y.to(device)
        with torch.no_grad():
            evidences, evidence_a = model(X, A_test, ce_scale=0)
            _, Y_pre = torch.max(evidence_a, dim=1)
            num_correct += (Y_pre == Y).sum().item()
            num_sample += Y.shape[0]

    # 计算准确率
    accuracy = 100 * num_correct / num_sample
    print(f'====> acc: {accuracy:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PIE', help='Dataset to use (Scene or PIE)')
    parser.add_argument('--model-path', type=str, default='pie_normal', help='Model to use (e.g., pie_normal)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train [default: 500]')
    parser.add_argument('--annealing_step', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR', help='learning rate')
    parser.add_argument('--gamma', type=float, default=1.0, metavar='G', help='gamma parameter for loss function')
    parser.add_argument('--neighbor', type=int, default=20, metavar='N',
                        help='number of neighbors for graph construction')
    parser.add_argument('--psi', type=float, default=0.7, metavar='P', help='psi parameter for model')
    parser.add_argument('--add-conflict', action='store_true', help='whether to test on conflict instances')
    args = parser.parse_args()
    train_and_evaluate(args)