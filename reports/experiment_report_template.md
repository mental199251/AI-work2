# 实验2 文本分类实验报告模板

## 一、实验目的
1. 理解 RNN/LSTM/BiLSTM/Attention 在文本分类中的机制与差异。
2. 掌握文本分类完整流程：预处理、建模、训练、评估、分析。

## 二、实验环境
- Python：
- 主要库：Numpy、Pandas、Scikit-learn、NLTK/Jieba、Matplotlib、PyTorch
- 硬件环境：

## 三、数据集与预处理
### 3.1 数据集说明
- 数据集名称：
- 类别数量：
- 样本总量：
- 划分比例：训练集 70%，验证集 10%，测试集 20%

### 3.2 预处理流程
- 文本清洗：
- 分词方法：
- 停用词处理：
- 向量化方式：Embedding
- 最大序列长度：
- 词表大小：

## 四、模型设计与参数设置
### 4.1 模型结构
- RNN：
- LSTM：
- BiLSTM：
- LSTM+Attention：

### 4.2 超参数
- Learning Rate：
- Batch Size：
- Epoch：
- Hidden Dim：
- Embedding Dim：
- Dropout：

## 五、实验结果
### 5.1 训练过程
插入训练曲线图（Loss / Accuracy）：
- RNN：
- LSTM：
- BiLSTM：
- LSTM+Attention：

### 5.2 测试集评估指标对比
| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| RNN |  |  |  |  |
| LSTM |  |  |  |  |
| BiLSTM |  |  |  |  |
| LSTM+Attention |  |  |  |  |

## 六、结果分析
### 6.1 RNN 与 LSTM 效果差异
- 对比结论：
- 原因分析（梯度消失、长依赖建模能力）：

### 6.2 BiLSTM 相比 LSTM 的优势
- 对比结论：
- 原因分析（前后文信息利用）：

### 6.3 Attention 机制带来的提升
- 对比结论：
- 原因分析（关键语义聚焦能力）：

### 6.4 问题与解决方法
- 过拟合表现与处理（正则化、早停、数据增强）：
- 梯度消失/训练不稳定处理（LSTM、学习率调整、梯度裁剪）：

## 七、实验总结
- 本次实验收获：
- 后续改进方向：
