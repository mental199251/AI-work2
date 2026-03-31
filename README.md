# 实验2 文本分类实验（RNN / LSTM / BiLSTM / LSTM+Attention）

本项目按照实验要求实现了完整文本分类流程：
- 数据预处理（清洗、分词、去停用词、向量化、7:1:2 划分）
- 四种模型（RNN、LSTM、BiLSTM、LSTM+Attention）
- 训练记录（loss/accuracy 曲线）
- 测试评估（accuracy/precision/recall/F1）
- 结果可视化与对比输出

## 1. 项目结构

```text
.
├── configs/
├── data/
│   ├── raw/
│   └── README.md
├── outputs/
│   ├── figures/
│   ├── history/
│   ├── metrics/
│   └── models/
├── reports/
│   └── experiment_report_template.md
├── src/
│   ├── data/
│   ├── models/
│   ├── trainers/
│   ├── utils/
│   ├── evaluate.py
│   └── main.py
└── requirements.txt
```

## 2. 环境安装

```bash
pip install -r requirements.txt
```

## 3. 数据准备

请参考 [data/README.md](data/README.md) 的格式放置数据。

支持两类数据：
1. `imdb`：CSV（`text`,`label`）或 ACLIMDB 文件夹结构
2. `thucnews`：类别子目录结构（每个子目录下为 `.txt`）

## 4. 运行方式

建议从项目根目录运行：

### 4.1 IMDB 示例

```bash
python -m src.main \
  --dataset imdb \
  --data_path data/raw/imdb.csv \
  --text_col text \
  --label_col label \
  --language en \
  --epochs 10 \
  --batch_size 64 \
  --max_len 300 \
  --models rnn,lstm,bilstm,lstm_attention
```

### 4.2 THUCNews 示例

```bash
python -m src.main \
  --dataset thucnews \
  --data_path data/raw/THUCNews \
  --language zh \
  --epochs 10 \
  --batch_size 64 \
  --max_len 500 \
  --models rnn,lstm,bilstm,lstm_attention
```

### 4.3 快速调试（每类抽样）

```bash
python -m src.main \
  --dataset thucnews \
  --data_path data/raw/THUCNews \
  --language zh \
  --max_samples_per_class 500 \
  --epochs 3
```

## 5. 输出结果

运行后将自动生成：
- `outputs/models/*.pt`：各模型最佳参数
- `outputs/history/*_history.json`：训练过程记录
- `outputs/figures/*_curves.png`：训练曲线（loss/accuracy）
- `outputs/figures/*_confusion_matrix.png`：混淆矩阵
- `outputs/metrics/model_comparison.csv`：模型评估指标对比
- `outputs/figures/model_comparison.png`：指标柱状对比图

## 6. 可调超参数

- 数据：`max_vocab_size`、`min_freq`、`max_len`、`stopwords_path`
- 模型：`embedding_dim`、`hidden_dim`、`dropout`
- 训练：`learning_rate`、`batch_size`、`epochs`、`early_stop_patience`

## 7. 报告撰写

实验报告可直接基于模板：
- [reports/experiment_report_template.md](reports/experiment_report_template.md)
