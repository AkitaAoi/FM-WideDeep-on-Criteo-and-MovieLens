import torch.nn as nn
from torch.nn.functional import dropout


class Wide(nn.Module):
    def __init__(self, dense_num, sparse_vocab_sizes, output_dim=1, l2_reg=1e-4):
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
    def __init__(self, sparse_vocab_sizes, embed_dim, output_dim, hidden_units, activation='relu', dropout_rate=0.3):
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

        self.dropout = nn.Dropout(p=dropout_rate)

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
            x = self.dropout(x)
        return self.output_layer(x)  # (batch, output_dim)

class WideDeep(nn.Module):
    def __init__(self, dense_num, sparse_vocab_sizes, embed_dim, output_dim, hidden_units, activation='relu',
                 dropout_rate=0.3):
        super(WideDeep, self).__init__()
        self.wide = Wide(dense_num, sparse_vocab_sizes, output_dim)
        self.deep = Deep(sparse_vocab_sizes, embed_dim, output_dim, hidden_units, activation, dropout_rate)

    def forward(self, dense_inputs, sparse_indices):
        wide_out = self.wide(dense_inputs, sparse_indices)
        deep_out = self.deep(sparse_indices)
        return torch.sigmoid(0.5 * (wide_out + deep_out))

import config
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_criteo_data(file_path, test_size=0.2, random_state=42):
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

def load_movielens_100k(data_path='.', test_size=0.2, random_state=42, threshold=3):
    # 1. 读取 u.data
    data = pd.read_csv(
        f'{data_path}/u.data',
        sep='\t',
        header=None,
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )

    # 2. 将评分转为二分类标签
    data['label'] = (data['rating'] >= threshold).astype(int)

    # 3. 获取用户数和电影数（ID从1开始，因此最大值就是数量）
    n_users = data['user_id'].max()  # 943
    n_items = data['item_id'].max()  # 1682
    n = n_users + n_items  # 2625

    # 4. 构建独热编码矩阵（稠密矩阵，2625列，内存可接受）
    # user_ids = data['user_id'].values - 1  # 转为0-based索引
    # item_ids = data['item_id'].values - 1
    # X = np.zeros((len(data), n), dtype=np.float32)
    # X[np.arange(len(data)), user_ids] = 1.0
    # X[np.arange(len(data)), n_users + item_ids] = 1.0

    # 新增一个虚拟数值特征（常数1）
    data['dummy'] = 1.0
    dense_features = ['dummy']
    sparse_features = ['user_id', 'item_id']

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

    print(f"数据加载完成：")
    print(f"  用户数: {n_users}, 电影数: {n_items}")
    print(f"  特征总数: {n}")
    print(f"  总样本数: {len(data)}")
    print(f"  正样本比例: {data['label'].mean():.2%}")
    print(f"  训练集: {X_sparse_train.shape[0]}, 测试集: {X_sparse_test.shape[0]}")
    print(f"sparse_vocab_sizes: {sparse_vocab_sizes}")

    return (X_dense_train, X_sparse_train, y_train), (X_dense_test, X_sparse_test, y_test), sparse_vocab_sizes

from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_movielens_with_context(data_path='.', test_size=0.2, random_state=42, threshold=3):
    # 1. 读取评分数据（u.data）
    ratings = pd.read_csv(
        f'{data_path}/u.data',
        sep='\t',
        header=None,
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )

    # 2. 读取用户数据（u.user）
    users = pd.read_csv(
        f'{data_path}/u.user',
        sep='|',
        header=None,
        names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
    )

    # 3. 读取电影数据（u.item）
    # 注意：u.item 有 24 列，前5列是电影ID、标题、上映日期、视频发布日期、IMDb链接，后面19列是类型标签
    item_columns = ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_url'] + [f'genre_{i}' for i in
                                                                                             range(19)]
    items = pd.read_csv(
        f'{data_path}/u.item',
        sep='|',
        header=None,
        encoding='latin-1',  # 有些电影名包含特殊字符
        names=item_columns
    )

    # 4. 合并数据（评分 + 用户特征 + 电影特征）
    data = ratings.merge(users, on='user_id').merge(items, on='item_id')

    # 5. 处理标签（二分类）
    data['label'] = (data['rating'] >= threshold).astype(int)

    # 6. 对用户ID和电影ID进行 LabelEncoder（用于one-hot索引）
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    data['user_encoded'] = user_encoder.fit_transform(data['user_id'])
    data['item_encoded'] = item_encoder.fit_transform(data['item_id'])

    # 7. 处理类别型上下文特征（用 LabelEncoder）
    gender_encoder = LabelEncoder()
    occ_encoder = LabelEncoder()
    data['gender_encoded'] = gender_encoder.fit_transform(data['gender'])
    data['occupation_encoded'] = occ_encoder.fit_transform(data['occupation'])

    # 9. 处理电影类型（19个类型标签，每个取值为0或1，本身就是one-hot）
    genre_cols = [f'genre_{i}' for i in range(19)]
    # 这19列已经是0/1，直接使用

    # 10. 构建特征矩阵 X
    # 特征组成：用户ID one-hot + 电影ID one-hot + 年龄(数值) + 性别编码 + 职业编码 + 19个类型标签
    n_users = len(user_encoder.classes_)  # 943
    n_items = len(item_encoder.classes_)  # 1682
    n_genders = len(gender_encoder.classes_)  # 2
    n_occupations = len(occ_encoder.classes_)  # 21

    dense_features = ['age']
    sparse_features = ['user_encoded', 'item_encoded', 'gender_encoded', 'occupation_encoded']

    # 词汇表大小直接使用之前统计的
    sparse_vocab_sizes = [n_users, n_items, n_genders, n_occupations]

    # ========== 关键修改：在提取特征之前，按时间戳排序 ==========
    data = data.sort_values('timestamp').reset_index(drop=True)

    # 然后分离特征和标签（从排序后的 data 中提取）
    X_dense = data[['age']].values.astype(np.float32)   # 注意这里用 'age' 原始值，暂不归一化
    X_sparse = data[['user_encoded', 'item_encoded', 'gender_encoded', 'occupation_encoded']].values.astype(np.int64)
    y = data['label'].values.astype(np.float32)

    # 按时间划分（前 80% 训练，后 20% 测试）
    split_idx = int(len(data) * (1 - test_size))
    X_dense_train, X_dense_test = X_dense[:split_idx], X_dense[split_idx:]
    X_sparse_train, X_sparse_test = X_sparse[:split_idx], X_sparse[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 8. 处理数值型上下文特征（归一化到0~1）, 避免数据泄露
    scaler = StandardScaler()
    X_dense_train = scaler.fit_transform(X_dense_train)
    X_dense_test = scaler.transform(X_dense_test)

    # 转为张量
    X_dense_train = torch.tensor(X_dense_train, dtype=torch.float32)
    X_sparse_train = torch.tensor(X_sparse_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_dense_test = torch.tensor(X_dense_test, dtype=torch.float32)
    X_sparse_test = torch.tensor(X_sparse_test, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    print(f"数据加载完成：")
    print(f"  用户数: {n_users}, 电影数: {n_items}")
    print(f"  特征总数: {sum(sparse_vocab_sizes) + 1}")
    print(f"  总样本数: {len(data)}")
    print(f"  正样本比例: {data['label'].mean():.2%}")
    print(f"  训练集: {X_sparse_train.shape[0]}, 测试集: {X_sparse_test.shape[0]}")
    print(f"sparse_vocab_sizes: {sparse_vocab_sizes}")

    return (X_dense_train, X_sparse_train, y_train), (X_dense_test, X_sparse_test, y_test), sparse_vocab_sizes

import utils
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def train(file_path):
    utils.set_seed(config.RANDOM_STATE)
    (X_dense_train, X_sparse_train, y_train), (X_dense_test, X_sparse_test,
                                               y_test), sparse_vocab_sizes = load_movielens_with_context(file_path)

    # ---- 新增：从训练集中划分验证集 (10%) ----
    X_dense_train, X_dense_val, X_sparse_train, X_sparse_val, y_train, y_val = train_test_split(
        X_dense_train, X_sparse_train, y_train, test_size=0.1, random_state=config.RANDOM_STATE)

    # DataLoader (训练集, 测试集)
    train_loader = utils.create_wideddeep_loader(X_dense_train, X_sparse_train, y_train, batch_size = config.BATCH_SIZE)
    test_loader = utils.create_wideddeep_loader(X_dense_test, X_sparse_test, y_test, batch_size = config.BATCH_SIZE)


    dense_num = X_dense_train.shape[1]

    model = WideDeep(
        dense_num=dense_num,
        sparse_vocab_sizes=sparse_vocab_sizes,
        embed_dim=config.WD_EMBED_DIM,
        output_dim=1,
        hidden_units=config.WD_HIDDEN_UNITS,
        activation='relu',
        dropout_rate=config.WD_DROPOUT
    )

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 训练
    losses = []
    epochs = 5
    for epoch in range(epochs):
        # 训练
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
        avg_train_loss = total_loss / len(train_loader)

        # ---- 验证集评估 ----
        model.eval()
        with torch.no_grad():
            val_pred = model(X_dense_val, X_sparse_val)
            val_loss = criterion(val_pred, y_val).item()
            val_auc = roc_auc_score(y_val.numpy(), val_pred.numpy())

        print(f"Epoch {epoch + 1}: Train Loss {avg_train_loss:.6f}, Val Loss {val_loss:.6f}, Val AUC {val_auc:.4f}")

    # 测试集评估
    test_auc, test_acc = utils.evaluate_binary(model, test_loader)
    print(f"Test AUC: {test_auc:.4f}, Test Acc: {test_acc:.4f}")

    # 作图
    # utils.plot_loss_curve(losses, val_losses=None, title='Training Loss')


from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


from torch.utils.data import DataLoader, TensorDataset

def cross_validate_widedeep(file_path, n_splits=5):
    # 1. 加载完整数据集（假设返回的是训练集部分，用于交叉验证）
    (X_dense, X_sparse, y), _, sparse_vocab_sizes = load_movielens_with_context(file_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_dense)):
        print(f"\n===== Fold {fold + 1}/{n_splits} =====")

        # 2. 分割数据
        X_dense_train, X_dense_val = X_dense[train_idx], X_dense[val_idx]
        X_sparse_train, X_sparse_val = X_sparse[train_idx], X_sparse[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 3. 创建 DataLoader（注意：DataLoader 需要 Dataset 对象）
        train_dataset = TensorDataset(X_dense_train, X_sparse_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

        # 4. 初始化模型（每折独立）
        model = WideDeep(
            dense_num=X_dense_train.shape[1],
            sparse_vocab_sizes=sparse_vocab_sizes,
            embed_dim=config.WD_EMBED_DIM,
            output_dim=1,
            hidden_units=config.WD_HIDDEN_UNITS,
            activation='relu'
        ).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        # 5. 训练模型
        epochs = 5  # 根据需要调整
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for dense_batch, sparse_batch, y_batch in train_loader:
                dense_batch = dense_batch.to(device)
                sparse_batch = sparse_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                pred = model(dense_batch, sparse_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

        # 6. 验证集评估（直接使用张量，无需重新包装）
        model.eval()
        with torch.no_grad():
            X_dense_val = X_dense_val.to(device)
            X_sparse_val = X_sparse_val.to(device)
            y_val_pred = model(X_dense_val, X_sparse_val).cpu().numpy().flatten()
            auc = roc_auc_score(y_val.cpu().numpy(), y_val_pred)  # y_val 是张量，需转为 numpy
            auc_scores.append(auc)
            print(f"  Fold {fold + 1} Val AUC: {auc:.4f}")

    # 7. 输出结果
    print(f"\n===== 交叉验证结果 =====")
    print(f"平均AUC: {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")
    print(f"各折AUC: {auc_scores}")
    return np.array(auc_scores)

if __name__ == '__main__':
    # train(file_path =config.MOVIELENS_DIR)
    cross_validate_widedeep(file_path=config.MOVIELENS_DIR)