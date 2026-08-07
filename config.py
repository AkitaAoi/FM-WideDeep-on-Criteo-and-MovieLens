# config.py
import os

# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Data')          # 存放 train.txt 和 ml-100k.zip

CRITEO_FILE = os.path.join(DATA_DIR, 'train.txt')
MOVIELENS_DIR = os.path.join(DATA_DIR, 'ml-100k')  # 解压后的 ml-100k 文件夹

# ========== 通用训练参数 ==========
RANDOM_STATE = 42
TEST_SIZE = 0.2
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.01

# ========== FM 参数 ==========
FM_EMBED_DIM = 32          # 对应 FM 的 k
FM_W_REG = 1e-4
FM_V_REG = 1e-4

# ========== WideDeep 参数 ==========
WD_EMBED_DIM = 4          # Deep 部分的嵌入维度
WD_HIDDEN_UNITS = [8, 4, 2]
WD_DROPOUT = 0.3
WD_L2_REG = 1e-4

# ========== MovieLens 数据处理参数 ==========
ML_THRESHOLD = 3            # 评分 ≥ 3 视为正样本
ML_USER_COL = 'user_id'
ML_ITEM_COL = 'item_id'