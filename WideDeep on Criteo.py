import torch.nn as nn

class Wide(nn.Module):
    def __init__(self, dense_num, sparse_vocab_sizes, output_dim=1):
        """
        dense_num: 稠密特征数量（如13）
        sparse_vocab_sizes: 每个稀疏特征的词汇表大小列表（长度等于稀疏特征个数）
        output_dim: 输出维度（1）
        l2_reg: L2正则化系数（仅对稠密线性层生效）
        """
        super(Wide, self).__init__()
        # 稠密特征部分：线性层
        self.dense_linear = nn.Linear(dense_num, output_dim, bias=True)
        # 稀疏特征部分：每个特征一个Embedding表，输出维度为1（即权重）
        self.sparse_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, 1) for vocab_size in sparse_vocab_sizes
        ])
        # 初始化
        nn.init.normal_(self.dense_linear.weight, mean=0, std=0.01)
        nn.init.zeros_(self.dense_linear.bias)
        for emb in self.sparse_embeddings:
            nn.init.normal_(emb.weight, mean=0, std=0.01)  # 初始化权重

    def forward(self, dense_inputs, sparse_indices):
        """
        dense_inputs: (batch, dense_num)
        sparse_indices: (batch, sparse_num) ，每列为整数索引
        """
        # 稠密部分输出
        dense_out = self.dense_linear(dense_inputs)  # (batch, 1)

        # 稀疏部分：查表并求和
        sparse_out = torch.zeros(dense_inputs.size(0), 1, device=dense_inputs.device)
        for i, emb in enumerate(self.sparse_embeddings):
            # emb(sparse_indices[:, i]) 形状 (batch, 1)
            sparse_out += emb(sparse_indices[:, i])

        return dense_out + sparse_out


class Deep(nn.Module):
    def __init__(self, sparse_vocab_sizes, embed_dim, output_dim, hidden_units, activation='relu'):
        """
        sparse_vocab_sizes: 每个稀疏特征的词汇表大小
        embed_dim: 每个特征的嵌入维度
        output_dim: 输出维度
        hidden_units: 隐藏层神经元列表，如 [256, 128]
        activation: 激活函数
        """
        super(Deep, self).__init__()
        # 为每个稀疏特征构建嵌入层
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim) for vocab_size in sparse_vocab_sizes
        ])
        # 拼接后的总维度
        deep_input_dim = len(sparse_vocab_sizes) * embed_dim

        # 构建隐藏层
        self.hidden_layers = nn.ModuleList()
        in_dim = deep_input_dim
        for unit in hidden_units:
            self.hidden_layers.append(nn.Linear(in_dim, unit))
            in_dim = unit
        self.output_layer = nn.Linear(in_dim, output_dim, bias=True)

        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Unsupported activation')

        # 初始化
        for layer in self.hidden_layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0, std=0.01)

    def forward(self, sparse_indices):
        """
        sparse_indices: (batch, sparse_num)
        """
        # 查表并拼接
        embed_list = [emb(sparse_indices[:, i]) for i, emb in enumerate(self.embeddings)]
        concat_embed = torch.cat(embed_list, dim=1)  # (batch, sparse_num * embed_dim)

        x = concat_embed
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)  # (batch, output_dim)

class WideDeep(nn.Module):
    def __init__(self, dense_num, sparse_vocab_sizes, embed_dim, output_dim, hidden_units, activation='relu'):
        super(WideDeep, self).__init__()
        self.wide = Wide(dense_num, sparse_vocab_sizes, output_dim)
        self.deep = Deep(sparse_vocab_sizes, embed_dim, output_dim, hidden_units, activation)

    def forward(self, dense_inputs, sparse_indices):
        wide_out = self.wide(dense_inputs, sparse_indices)
        deep_out = self.deep(sparse_indices)
        return torch.sigmoid(0.5 * (wide_out + deep_out))

import config
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_criteo_data(file_path, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE):
    """
    加载Criteo数据，返回：
    - 稠密特征矩阵 (样本数, 13)
    - 稀疏特征索引矩阵 (样本数, 26)，每列为整数编码
    - 标签向量
    - 每个稀疏特征的词汇表大小列表（用于模型初始化）
    """
    data = pd.read_csv(file_path)

    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]

    # 填充缺失值
    data[dense_features] = data[dense_features].fillna(0)
    data[sparse_features] = data[sparse_features].fillna('-1')

    # 归一化数值特征
    scaler = MinMaxScaler()
    data[dense_features] = scaler.fit_transform(data[dense_features])

    # 对稀疏特征进行整数编码（Label Encoding）
    sparse_vocab_sizes = []
    for col in sparse_features:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        sparse_vocab_sizes.append(len(le.classes_))  # 记录词汇表大小

    # 分离特征和标签
    X_dense = data[dense_features].values.astype(np.float32)
    X_sparse = data[sparse_features].values.astype(np.int64)  # 整数索引
    y = data['label'].values.astype(np.float32)

    # 划分训练集和测试集
    X_dense_train, X_dense_test, X_sparse_train, X_sparse_test, y_train, y_test = train_test_split(
        X_dense, X_sparse, y, test_size=test_size, random_state=random_state
    )

    # 转为张量
    X_dense_train = torch.tensor(X_dense_train, dtype=torch.float32)
    X_sparse_train = torch.tensor(X_sparse_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_dense_test = torch.tensor(X_dense_test, dtype=torch.float32)
    X_sparse_test = torch.tensor(X_sparse_test, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    return (X_dense_train, X_sparse_train, y_train), (X_dense_test, X_sparse_test, y_test), sparse_vocab_sizes

import utils
import torch
import torch.nn as nn

def train(file_path):
    utils.set_seed(config.RANDOM_STATE)
    (X_dense_train, X_sparse_train, y_train), (X_dense_test, X_sparse_test,
                                               y_test), sparse_vocab_sizes = load_criteo_data(file_path)

    # DataLoader
    train_loader = utils.create_wideddeep_loader(X_dense_train, X_sparse_train, y_train, batch_size = config.BATCH_SIZE)
    test_loader = utils.create_wideddeep_loader(X_dense_test, X_sparse_test, y_test, batch_size = config.BATCH_SIZE)

    # 模型
    model = WideDeep(
        dense_num=13,
        sparse_vocab_sizes=sparse_vocab_sizes,
        embed_dim=config.WD_EMBED_DIM,
        output_dim=1,
        hidden_units=config.WD_HIDDEN_UNITS,
        activation='relu'
    )

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练
    losses = []
    epochs = config.EPOCHS
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for dense_batch, sparse_batch, y_batch in train_loader:
            pred = model(dense_batch, sparse_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.6f}')

    # 评估
    test_auc, test_acc = utils.evaluate_binary(model, test_loader)
    print(f"Test AUC: {test_auc:.4f}, Test Acc: {test_acc:.4f}")

    # 作图
    utils.plot_loss_curve(losses, val_losses=None, title='Training Loss')

if __name__ == '__main__':
    train(file_path = config.CRITEO_FILE)
