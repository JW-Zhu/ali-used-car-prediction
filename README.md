#  Used Car Price Prediction

本项目基于 [阿里云天池 - 二手车交易价格预测比赛](https://tianchi.aliyun.com/competition/entrance/231784/rankRange)，赛题任务是预测二手车的交易价格。该比赛属于天池学习赛，我在 **第三期学习赛中荣获冠军 🏆**。

---

##  比赛背景

二手车市场规模庞大，而如何合理定价是一个关键问题。  
本赛题提供了超过 **40 万条二手车交易记录**，数据集来源于某二手车交易平台。  
数据集包含 **31 列特征**，其中包括：  
- 公开字段（如 `model`、`brand`、`regionCode` 等，已进行脱敏处理）  
- 匿名变量（共 15 列）  

为了保证公平性，主办方将数据划分为：  
- **训练集**：15 万条样本  
- **测试集 A**：5 万条样本  
- **测试集 B**：5 万条样本  

参赛选手需基于训练数据构建模型，对测试集的二手车价格进行预测，并通过排行榜进行评估。

---

##  数据说明

- 数据集需在天池官网报名后下载。  
- 部分字段已进行脱敏处理（如 `name`、`model`、`brand`、`regionCode` 等）。  
- 特征类型多样，包含数值型、类别型以及匿名特征。  

---

## 📂 项目结构
```
car/
├── data/ # 数据目录（未上传）
├── notebooks/ # Jupyter 笔记本
│ └── eda.ipynb # 探索性数据分析
├── src/ # 源代码
│ ├── data_preprocessing.py
│ └── feature_engineering.py
├── main.ipynb # 主流程 Notebook
├── requirements.txt # 依赖文件
└── README.md # 项目说明
```
## 项目模型
本项目最终采用 深度学习神经网络 + 注意力机制 (Attention) 进行预测建模。

模型设计思路

特征表征

使用多层全连接网络 (MLP) 对输入特征进行非线性映射。

每层引入 Batch Normalization，加快收敛并提升稳定性。

注意力机制 (Attention)

在第一层隐空间中加入自定义 Attention 模块。

模块通过两层全连接网络生成特征权重 (0~1)，动态调整不同特征维度的重要性。

相当于在网络内部实现了「自动特征选择」，提升模型表现。

正则化与优化

Dropout：减少过拟合，提高泛化能力。

K-Fold 交叉验证：提升结果稳健性。

Adam 优化器 作为训练算法。

损失函数为 MAE (Mean Absolute Error)，与比赛评估指标保持一致。

模型结构简述

输入层：input_dim

隐藏层：

Dense(300) → BatchNorm → Attention → Dropout

Dense(300) → BatchNorm → Dropout

Dense(64) → Dense(32) → Dense(8)

输出层：Dense(1)
