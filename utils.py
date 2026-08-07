# utils.py
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score

def set_seed(seed=42):
    """固定所有随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def plot_loss_curve(train_losses, val_losses=None, title='Training Loss'):
    """绘制训练（和验证）损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def save_model(model, path):
    """保存模型权重"""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """加载模型权重"""
    model.load_state_dict(torch.load(path))
    return model

def evaluate_binary(model, data_loader, device='cpu'):
    """在二分类任务上计算 AUC 和准确率"""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            # 假设 data_loader 返回 (dense, sparse, label) 或 (X, label) 需要根据你的实际情况调整
            if len(batch) == 3:
                dense, sparse, label = batch
                dense = dense.to(device)
                sparse = sparse.to(device)
                preds = model(dense, sparse)
            else:
                X, label = batch
                X = X.to(device)
                preds = model(X)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(label.cpu().numpy().flatten())
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, (np.array(all_preds) >= 0.5).astype(int))
    return auc, acc



def create_wideddeep_loader(dense_data, sparse_data, labels, batch_size, shuffle=True):
    """
    为 WideDeep 模型创建 DataLoader，返回 (dense, sparse, label)
    """
    dataset = TensorDataset(dense_data, sparse_data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def sample_negative_data_fm(X, y, neg_ratio=3, random_state=42):
    """
    对 FM 的训练集进行负采样。
    X: 特征矩阵 (样本数, 特征数)
    y: 标签 (样本数,)
    返回采样后的 (X, y)
    """
    np.random.seed(random_state)

    # 确保 y 是 numpy 数组
    if hasattr(y, 'values'):
        y = y.values
    if hasattr(X, 'values'):
        X = X.values

    pos_mask = (y == 1)
    neg_mask = (y == 0)

    X_pos = X[pos_mask]
    y_pos = y[pos_mask]

    X_neg = X[neg_mask]
    y_neg = y[neg_mask]  # 现在 y_neg 已经是 numpy 数组

    n_pos = len(y_pos)
    n_neg_target = int(n_pos * neg_ratio)

    if len(y_neg) > n_neg_target:
        indices = np.random.choice(len(y_neg), n_neg_target, replace=False)
        X_neg = X_neg[indices]
        y_neg = y_neg[indices]  # numpy 数组支持位置索引

    X_sampled = np.concatenate([X_pos, X_neg], axis=0)
    y_sampled = np.concatenate([y_pos, y_neg], axis=0)

    shuffle_idx = np.random.permutation(len(y_sampled))
    X_sampled = X_sampled[shuffle_idx]
    y_sampled = y_sampled[shuffle_idx]

    print(f"负采样后：正样本 {n_pos}，负样本 {len(y_neg)}，比例 1:{len(y_neg) / n_pos:.1f}")
    return X_sampled, y_sampled


def to_tensor(data, dtype=torch.float32, is_label=False):
    """
    将数据转换为 torch.Tensor。
    - data: 可以是 numpy ndarray, pandas Series, pandas DataFrame 或 list
    - dtype: 目标数据类型（默认 torch.float32）
    - is_label: 如果为 True，则将张量形状调整为 (-1, 1)
    返回: torch.Tensor
    """
    # 如果 data 是 pandas 对象，提取 .values
    if hasattr(data, 'values'):
        data = data.values
    # 如果还不是 numpy 数组，强制转换
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    # 转为张量
    tensor = torch.tensor(data, dtype=dtype)
    # 如果是标签，确保形状为 (batch, 1)
    if is_label:
        tensor = tensor.view(-1, 1)
    return tensor