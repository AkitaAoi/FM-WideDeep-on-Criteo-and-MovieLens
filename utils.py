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

def create_fm_loader(features, labels, batch_size, shuffle=True):
    """
    为 FM 模型创建 DataLoader，返回 (X, label)
    """
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)