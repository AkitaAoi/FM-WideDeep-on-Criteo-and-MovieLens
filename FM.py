import torch.nn as nn

class FMs(nn.Module):
    def __init__(self, n, k, w_reg=0.0, v_reg=0.0):
        """
        n: 输入特征的总维度（即特征向量的长度）
           例如 Criteo 数据集中为 13（稠密）+ 26（稀疏）= 39
        k: 隐向量 vi 的维度，超参数。
           对应 FM 公式中二阶交互项的因子分解维度。
        w_reg: 一阶线性部分（w）的 L2 正则化系数
        v_reg: 二阶交互部分（v）的 L2 正则化系数
        """
        super(FMs, self).__init__()
        self.n = n
        self.k = k

        # 偏置项（全局偏差）
        self.w0 = nn.Parameter(torch.zeros(1))
        # 一阶特征权重（对应公式中的 w_i）
        self.w = nn.Parameter(torch.randn(n, 1))
        # 二阶交互的隐向量（对应公式中的 v_i）
        self.v = nn.Parameter(torch.randn(n, k))

        # 初始化
        nn.init.normal_(self.w, mean=0, std=0.1)
        nn.init.normal_(self.v, mean=0, std=0.1)

        self.w_reg = w_reg
        self.v_reg = v_reg

    def forward(self, x):
        if x.dim() != 2:
            raise ValueError('Input tensor needs to be a 2D tensor')

        # 线性部分：w0 + Σ w_i * x_i
        linear_part = self.w0 + torch.mm(x, self.w)

        # 二阶交互部分（利用公式简化计算，避免 O(n^2) 复杂度）
        # 公式：1/2 * Σ_f ( (Σ_i v_i,f x_i)^2 - Σ_i (v_i,f x_i)^2 )
        inter_part1 = torch.mm(x, self.v) ** 2
        inter_part2 = torch.mm(x ** 2, self.v ** 2)
        inter_part = 0.5 * torch.sum(inter_part1 - inter_part2, dim=1, keepdim=True)

        # 最终预测（logit + sigmoid）
        y = linear_part + inter_part
        return torch.sigmoid(y)

    def regularization(self):
        # L2 正则化项（不含偏置 w0，因为偏置通常不需要正则化）
        w_p = self.w_reg * torch.sum(self.w ** 2)
        v_p = self.v_reg * torch.sum(self.v ** 2)
        return w_p + v_p

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import config

def read_criteo(file_path, test_size = 0.3, neg_sample_ratio = 3):
    data = pd.read_csv(file_path)

    dense_feature = ['I' + str(i) for i in range(1, 14)]
    sparse_feature = ['C' + str(i) for i in range(1, 27)]

    # 填充缺失值
    data[dense_feature] = data[dense_feature].fillna(0)
    data[sparse_feature] = data[sparse_feature].fillna('-1')

    # 数值特征归一化，类别特征独热化
    data[dense_feature] = MinMaxScaler().fit_transform(data[dense_feature])
    data = pd.get_dummies(data)

    # 分离特征和标签
    x = data.drop(['label'], axis = 1).values
    y = data['label']

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size)

    # 负采样（只在训练集）
    if neg_sample_ratio > 0:
        x_train, y_train = utils.sample_negative_data_fm(x_train, y_train, neg_ratio=neg_sample_ratio, random_state=config.RANDOM_STATE)

    # 转为张量
    x_train = utils.to_tensor(x_train, dtype=torch.float32)
    y_train = utils.to_tensor(y_train, dtype=torch.float32, is_label=True)
    x_test = utils.to_tensor(x_test, dtype=torch.float32)
    y_test = utils.to_tensor(y_test, dtype=torch.float32, is_label=True)

    return (x_train, y_train), (x_test, y_test)


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
    user_ids = data['user_id'].values - 1  # 转为0-based索引
    item_ids = data['item_id'].values - 1
    X = np.zeros((len(data), n), dtype=np.float32)
    X[np.arange(len(data)), user_ids] = 1.0
    X[np.arange(len(data)), n_users + item_ids] = 1.0

    y = data['label'].values.astype(np.float32)

    # 5. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 6. 转为 PyTorch Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    print(f"数据加载完成：")
    print(f"  用户数: {n_users}, 电影数: {n_items}")
    print(f"  特征总数: {n}")
    print(f"  总样本数: {len(data)}")
    print(f"  正样本比例: {data['label'].mean():.2%}")
    print(f"  训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

    return (X_train, y_train), (X_test, y_test), n


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

    # 8. 处理数值型上下文特征（归一化到0~1）
    scaler = StandardScaler()
    data['age_norm'] = scaler.fit_transform(data[['age']])  # 注：年龄范围约7~73，归一化后可提高收敛速度

    # 9. 处理电影类型（19个类型标签，每个取值为0或1，本身就是one-hot）
    genre_cols = [f'genre_{i}' for i in range(19)]
    # 这19列已经是0/1，直接使用

    # 10. 构建特征矩阵 X
    # 特征组成：用户ID one-hot + 电影ID one-hot + 年龄(数值) + 性别编码 + 职业编码 + 19个类型标签
    n_users = len(user_encoder.classes_)  # 943
    n_items = len(item_encoder.classes_)  # 1682
    n_genders = len(gender_encoder.classes_)  # 2
    n_occupations = len(occ_encoder.classes_)  # 21

    # 先创建独热编码部分（用户ID + 电影ID）
    user_ids = data['user_encoded'].values
    item_ids = data['item_encoded'].values
    n_onehot = n_users + n_items
    X_onehot = np.zeros((len(data), n_onehot), dtype=np.float32)
    X_onehot[np.arange(len(data)), user_ids] = 1.0
    X_onehot[np.arange(len(data)), n_users + item_ids] = 1.0

    # 再添加其他上下文特征
    X_context = np.column_stack([
        data['age_norm'].values.astype(np.float32),  # 1列，归一化后的年龄
        data['gender_encoded'].values.astype(np.float32),  # 1列，性别编码
        data['occupation_encoded'].values.astype(np.float32),  # 1列，职业编码
        data[genre_cols].values.astype(np.float32)  # 19列，电影类型
    ])

    # 拼接所有特征
    X = np.concatenate([X_onehot, X_context], axis=1)

    # 注意：n 现在是一阶和二阶特征的总数
    n = X.shape[1]

    y = data['label'].values.astype(np.float32)

    data = data.sort_values('timestamp').reset_index(drop=True)

    # 按时间划分（前 80% 训练，后 20% 测试）
    split_idx = int(len(data) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]


    # 转为 PyTorch Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    print(f"  用户数: {n_users}, 电影数: {n_items}")
    print(f"  特征总数: {n}")
    print(f"  总样本数: {len(data)}")
    print(f"  正样本比例: {data['label'].mean():.2%}")
    print(f"  训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

    return (X_train, y_train), (X_test, y_test), n

import torch
import numpy as np
import utils
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

def train(file_path, test_size):
    utils.set_seed(config.RANDOM_STATE)
    # MovieLens数据集
    # (x_train, y_train), (x_test, y_test), n = load_movielens_with_context(file_path, test_size)
    # k = config.FM_EMBED_DIM  # 隐向量维度

    # criteo数据集
    (x_train, y_train), (x_test, y_test) = read_criteo(file_path, test_size = test_size)
    n = x_train.shape[1]
    k = 2

    # 模型
    model = FMs(
        n = n,
        k = k,
        w_reg = config.FM_W_REG,
        v_reg = config.FM_V_REG
    )
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # 训练
    losses = []
    epochs = 100
    for epoch in range(epochs):
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())  # 记录损失
        print(f'Epoch {epoch}: Loss {loss.item():.4f}')

    with torch.no_grad():
        y_pred = model(x_test)
        auc = roc_auc_score(y_test.numpy(), y_pred.numpy())
        print(f"AUC: {auc:.4f}")

    # 评估
    with torch.no_grad():
        y_test_pred = model(x_test)
        y_pred_binary = (y_test_pred >= 0.5).float().numpy().flatten()
        y_test_np = y_test.numpy().flatten()
        acc = accuracy_score(y_pred_binary, y_test_np)
        print(f'Accuracy {acc:.4f}')

    # 作图
    utils.plot_loss_curve(losses, val_losses=None, title='Training Loss')



if __name__ == '__main__':
    train(file_path =config.CRITEO_FILE, test_size= config.TEST_SIZE)
