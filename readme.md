## 开发文档

### 项目概述

该项目旨在通过加权 Jaccard 和 SequenceMatcher 相似度算法，自动化比较公司名称及其相关区域信息，以帮助识别相似的公司名称。项目主要功能包括加载数据、预处理文本、计算相似度以及保存结果。

### 项目目录结构

```
/project-root
│
├── main.py                  # 主程序文件
├── company.xlsx             # 输入的公司数据 Excel 文件
├── region_dict.txt          # 可选的区域词典文本文件
└── README.md                # 项目的说明文档
```

### 技术栈

- **Python**: 编程语言
- **Pandas**: 数据处理和分析
- **NumPy**: 数值计算
- **Jieba**: 中文分词
- **TQDM**: 进度条显示
- **Concurrent Futures**: 实现多进程处理
- **Openpyxl**: 读取和写入 Excel 文件

### 安装依赖

在项目根目录下，使用以下命令安装依赖：

```bash
pip install pandas numpy jieba tqdm openpyxl
```

### 准备输入数据

**创建 Excel 文件:**

- 创建一个名为 `company.xlsx` 的 Excel 文件。
- 在第一个工作表（Sheet1）中，放入第一个公司数据集，包含以下列：
    - `id`
    - `company_name`
- 在第二个工作表（Sheet2）中，放入第二个公司数据集，列结构与 Sheet1 相同。

**可选区域词典:**

- 创建一个名为 `region_dict.txt` 的文本文件，用于自定义区域词典。每行代表一个区域。

### 配置脚本参数

在 `main.py` 中，配置以下参数：

- `file_path`: 包含输入 Excel 文件和区域词典的目录路径。
- `jaccard_threshold`: 加权 Jaccard 相似度的阈值，范围为 0 到 1。
- `SequenceMatcher_threshold`: 加权 SequenceMatcher 相似度的阈值，范围为 0 到 1。
- `similarity_type`:
    - `'01'`：使用加权 Jaccard 相似度。
    - `'10'`：使用加权 SequenceMatcher 相似度。
    - `'11'`：使用均衡方法选择更好的相似度分数。
- `exact_match_score`: 分配给公司名称和区域完全匹配的分数。
- `max_workers`: 用于多进程处理的进程数。根据您的 CPU 核心数进行调整。
- `batch_size`: 每个进程一次处理的记录数。

### 输出

该脚本将在输入 Excel 文件中创建一个名为“SimilarityResults”的新工作表。该工作表将包含匹配结果，并包含以下列：

- `id1`: 第一个工作表中公司的 ID。
- `Company_Name1`: 第一个工作表中的公司名称。
- `similarity`: 计算出的相似度分数。
- `similarity_type`: 使用的相似度类型（01 表示 Jaccard，10 表示 SequenceMatcher）。
- `Company1`: 来自第一个工作表的规范化公司名称。
- `Area1`: 从第一个工作表中的公司名称中提取的区域。
- `BestMatchID`: 第二个工作表中最匹配公司的 ID。
- `BestMatchCompany_Name`: 第二个工作表中最匹配公司的公司名称。
- `BestMatchCompany`: 来自第二个工作表的规范化公司名称。
- `BestMatchArea`: 从第二个工作表中最匹配公司的公司名称中提取的区域。

### 注意事项

- 为了获得最佳性能，请确保 `jieba` 库使用自定义区域词典正确配置。
- 根据您的数据和所需的匹配精度调整相似度阈值和批处理大小。

### 主要模块

#### 1. 数据加载与预处理

- `load_region_dict(dict_path)`
    - 功能: 加载区域字典并返回区域集合。
    - 参数: `dict_path`: 字典文件路径。
    - 返回: 区域集合。

- `load_and_preprocess_data(input_file_path, dict_path)`
    - 功能: 从 Excel 文件加载数据，并进行预处理。
    - 参数:
        - `input_file_path`: 输入 Excel 文件路径。
        - `dict_path`: 区域字典文件路径。
    - 返回: 预处理后的两个数据框（df1 和 df2）。

#### 2. 文本处理

- `preprocess_text(text)`
    - 功能: 对文本进行分词处理。
    - 参数: `text`: 待处理文本。
    - 返回: 分词后的列表。

- `preprocess_company_area(company_name, region_set)`
    - 功能: 处理公司名称，提取公司名和区域信息。
    - 参数:
        - `company_name`: 公司名称字符串。
        - `region_set`: 区域集合。
    - 返回: 处理后的公司名称和区域。

#### 3. 相似度计算

- `weighted_jaccard_similarity(tokens1, tokens2)`
    - 功能: 计算加权 Jaccard 相似度。
    - 参数:
        - `tokens1`: 第一个公司名称分词列表。
        - `tokens2`: 第二个公司名称分词列表。
    - 返回: Jaccard 相似度值。

- `weighted_SequenceMatcher_similarity(tokens1, tokens2)`
    - 功能: 计算加权SequenceMatcher相似度。
    - 参数:
        - `tokens1`: 第一个公司名称分词列表。
        - `tokens2`: 第二个公司名称分词列表。
    - 返回: 莱温斯坦相似度值。

#### 4. 匹配与结果保存

- `find_matches(df1, df2, jaccard_threshold, SequenceMatcher_threshold, similarity_type, exact_match_score, num_workers=1, batch_size=10)`
    - 功能: 在两个数据框中查找匹配的公司。
    - 参数:
        - `df1`: 第一个数据框。
        - `df2`: 第二个数据框。
        - `jaccard_threshold`: Jaccard 相似度阈值。
        - `SequenceMatcher_threshold`: SequenceMatcher 相似度阈值。
        - `similarity_type`: 使用的相似度类型。
        - `exact_match_score`: 完全匹配的分数。
        - `num_workers`: 使用的进程数。
        - `batch_size`: 每次处理的记录数。
    - 返回: 匹配结果的数据框。

- `save_results(matches_df, output_file)`
    - 功能: 保存匹配结果到 Excel 文件。
    - 参数:
        - `matches_df`: 匹配结果数据框。
        - `output_file`: 输出文件路径。
    - 返回: None

### 使用说明

1. 准备数据: 确保 `company.xlsx` 和 `region_dict.txt` 文件在项目根目录下。
2. 设置参数: 根据需要调整 `main.py` 中的参数设置，例如阈值、相似度类型等。
3. 运行程序: 在命令行中执行以下命令： `python main.py`
4. 查看结果: 结果将保存在输入文件同一位置的 `SimilarityResults` 工作表中。

### 未来计划

- 增加对更多相似度算法的支持。
- 提供可视化工具以更直观地展示匹配结果。
- 优化性能以提高处理速度。

