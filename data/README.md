# 数据放置说明

## 1) IMDB（基础版）

支持两种格式：

### 格式A：CSV
示例路径：`data/raw/imdb.csv`

CSV 至少包含两列：
- `text`: 评论文本
- `label`: 标签（如 `positive/negative` 或 `0/1`）

### 格式B：ACLIMDB 原始目录
示例路径：`data/raw/aclImdb`

目录结构示例：
```text
aclImdb/
├── train/
│   ├── pos/*.txt
│   └── neg/*.txt
└── test/
    ├── pos/*.txt
    └── neg/*.txt
```

程序会读取 train+test 后重新按 7:1:2 划分。

## 2) THUCNews（进阶版）

示例路径：`data/raw/THUCNews`

目录结构示例：
```text
THUCNews/
├── 体育/
│   ├── 1.txt
│   ├── 2.txt
│   └── ...
├── 财经/
├── 娱乐/
└── 科技/
```

程序会自动将子目录名作为类别名，并按 7:1:2 划分。

## 3) 停用词（可选）

可提供一个停用词文件（每行一个词）：
- 英文：`data/raw/stopwords_en.txt`
- 中文：`data/raw/stopwords_zh.txt`

运行时通过 `--stopwords_path` 指定。
