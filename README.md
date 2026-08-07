# FM-WideDeep-on-Criteo-and-MovieLens
## 项目简介：
使用FM算法与WideDeep算法分别在Criteo与MovieLens数据集上进行预测，得到基线模型.

## 项目结构

```
FM-WideDeep-on-Criteo-and-MovieLens/
├── Data/
│   ├── train.txt                # Criteo 数据集
│   └── ml-100k/                 # MovieLens 数据集
│       ├── u.data
│       ├── u.user
│       └── u.item
├── FM.py                        # FM 模型训练脚本
├── WideDeep on Criteo.py       # WideDeep 在 Criteo 上的训练
├── WideDeep on MovieLens.py    # WideDeep 在 MovieLens 上的训练
├── utils.py                     # 辅助函数
├── config.py                    # 配置文件
├── requirements.txt             # 依赖列表
├── README.md
└── .gitignore
```


## 数据集说明：
Criteo使用train.txt, MovieLens使用ml-100k数据集.


## 模型流程图：
#### FM：
```mermaid
flowchart LR
    subgraph 输入
        A[输入 X batch x n]
    end

    subgraph 线性部分
        A --> B[线性变换 w0 + sum wi*xi]
        B --> C[linear_part batch x 1]
    end

    subgraph 二阶交互部分
        A --> D[计算 sum vi,f * xi]
        D --> E[求平方]
        A --> F[计算 sum vi,f^2 * xi^2]
        F --> G[对因子求和]
        E --> H[两项相减]
        G --> H
        H --> I[乘以 0.5]
        I --> J[inter_part batch x 1]
    end

    subgraph 输出
        C --> K[相加 linear + inter]
        J --> K
        K --> L[Sigmoid]
        L --> M[y_pred batch x 1]
    end
```

#### WideDeep：
```mermaid
flowchart LR
    A[输入 batch, 13+26] --> B[分离特征]

    B --> C1[X_dense batch, 13]
    B --> C2[X_sparse batch, 26]

    subgraph Wide 分支
        C1 --> D1[线性层 dense_linear]
        D1 --> E1[wide_output batch, 1]

        C2 --> D2[Embedding vocab_size, 1 查表取权重]
        D2 --> D3[相加所有权重]
        D3 --> E1
    end

    subgraph Deep 分支
        C2 --> F1[Embedding vocab_size, embed_dim 查表取向量]
        F1 --> F2[拼接所有向量 batch, 26*embed_dim]
        F2 --> F3[全连接网络 MLP 隐藏层+激活]
        F3 --> E2[deep_output batch, 1]
    end

    E1 --> G[wide_output + deep_output]
    E2 --> G
    G --> H[sigmoid]
    H --> I[预测概率 y_pred]
```
## 运行方式：
pip install -r requirements.txt + python FM.py + WideDeep on Criteo.py + WideDeep on MoiveLens.py

## 代码结构说明:

FM.py：包含 FM 模型定义、Criteo 和 MovieLens 数据的加载与训练逻辑。

WideDeep on Criteo.py：包含 Wide & Deep 模型在 Criteo 数据集上的训练与评估。

WideDeep on MoiveLens.py：包含 Wide & Deep 模型在 MovieLens 数据集上的训练与评估

## 运行结果：

| 模型 | 数据集 | 主要指标 | 结果     |
| ------ | :-------: |------------- |--------|
| FM | Criteo | 准确率 | 0.7867 |
| FM | ml-100k | AUC | 0.7526 |
| WideDeep | Criteo | 准确率 | 0.7775 |
| WideDeep | ml-100k | AUC | 0.8009 |


FM on Criteo(Acc = 0.7867):

<img width="640" height="480" alt="Figure_2" src="https://github.com/user-attachments/assets/510bca2f-b5df-4626-94d7-eb8ff439598b" />

FM on MovieLens(k = 32, AUC = 0.7526):

<img width="640" height="480" alt="k = 32" src="https://github.com/user-attachments/assets/e23982cc-e9f6-4c35-bbf6-1be7177f78bf" />

WideDeep on Criteo(Acc = 0.7775):

<img width="467" height="168" alt="image" src="https://github.com/user-attachments/assets/b91f3104-815a-4198-b2d8-8f764df9c2ca" />

WideDeep on MovieLens(AUC = 0.8009):

<img width="957" height="357" alt="运行结果" src="https://github.com/user-attachments/assets/59730d88-cd57-48c1-bbf7-7222e1c62f25" />

## 核心结果解读
WideDeep在MovieLens上AUC达0.8010，较FM提升约0.05，说明引入深度网络能更好地捕捉特征交互。

## 局限与改进方向
当前未做超参数调优，未来可使用Optuna搜索；模型未在完整Criteo数据集上验证，后续可扩展

基于FM的CTR预测: [https://blog.csdn.net/Amarashi/article/details/163417299?spm=1011.2415.3001.5331]

基于WideDeep的CTR预测: [https://blog.csdn.net/Amarashi/article/details/163399394?spm=1011.2415.3001.5331]

基于FM的MovieLens的评分预测: [https://blog.csdn.net/Amarashi/article/details/163483814?spm=1011.2415.3001.5331]

基于WideDeep的MovieLens的评分预测:[https://blog.csdn.net/Amarashi/article/details/163542955spm=1011.2415.3001.5331]
