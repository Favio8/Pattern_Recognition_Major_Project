# KNN分类算法实现与实验

## 项目概述

本项目是西安电子科技大学模式识别课程第一次大作业，实现了K近邻(K-Nearest Neighbors, KNN)分类算法，并在三个经典数据集上进行了实验验证。项目包含完整的算法实现、多种距离度量方法、不同K值的性能比较以及详细的实验分析。

## 数据集

### 1. USPS手写体数字识别数据集

- **数据规模**: 9,298个样本，256个特征
- **类别数**: 10类 (数字0-9)
- **特征描述**: 16×16像素的灰度图像，展平为256维向量
- **文件**: `KNN_USPS_Classification.ipynb`

### 2. UCI Sonar数据集

- **数据规模**: 208个样本，60个特征
- **类别数**: 2类 (Mine/Rock - 水雷/岩石)
- **特征描述**: 声纳信号在不同角度的能量反射值
- **数据文件**: `connectionist+bench+sonar+mines+vs+rocks/`
- **实验文件**: `KNN_Sonar_Classification.ipynb`

### 3. UCI Iris鸢尾花数据集

- **数据规模**: 150个样本，4个特征
- **类别数**: 3类 (Setosa, Versicolor, Virginica)
- **特征描述**: 花萼长度、花萼宽度、花瓣长度、花瓣宽度
- **文件**: `KNN_Iris_Classification.ipynb`

## 算法实现

### KNN分类器特性

- **自定义实现**: 从零开始实现KNN算法，不依赖sklearn的KNN
- **多种距离度量**:
  - 欧几里得距离 (Euclidean Distance)
  - 曼哈顿距离 (Manhattan Distance)  
  - 余弦距离 (Cosine Distance)
- **灵活的K值设置**: 支持任意K值的设置和比较

### 核心功能

```python
class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean')
    def fit(X_train, y_train)
    def predict(X_test)
    def predict_single(x)
```

## 实验设计

### 验证算法

1. **K近邻分类**: 基础KNN算法实现
2. **最近邻分类**: K=1的特殊情况
3. **参数分析**: 不同K值和距离度量的性能比较

### 评估指标

- **OA (Overall Accuracy)**: 总体准确率
- **AA (Average Accuracy)**: 平均准确率
- **Kappa系数**: Cohen's Kappa统计量
- **混淆矩阵**: 详细的分类结果分析

### 实验方法

- **训练集比例分析**: 不同训练样本比例的性能影响
- **K折交叉验证**: 5折交叉验证确保结果可靠性
- **可视化分析**: 错误率/正确率曲线图

## 项目结构

```
KNN/
├── KNN_Iris_Classification.ipynb      # Iris数据集实验
├── KNN_Sonar_Classification.ipynb     # Sonar数据集实验  
├── KNN_USPS_Classification.ipynb      # USPS数据集实验
├── connectionist+bench+sonar+mines+vs+rocks/  # Sonar数据文件
│   ├── sonar.all-data
│   ├── sonar.mines
│   ├── sonar.rocks
│   └── sonar.names
├── README.md                          # 项目说明文档
└── 作业要求.png                       # 作业要求说明
```

## 运行环境

### 依赖库

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
```

### 环境配置

- Python 3.7+
- Jupyter Notebook
- 中文字体支持 (SimHei)

## 使用方法

### 1. 运行单个数据集实验

```bash
# 打开对应的Jupyter Notebook文件
jupyter notebook KNN_Iris_Classification.ipynb
```

### 2. 主要实验步骤

1. **数据加载与预处理**
2. **KNN分类器训练**
3. **不同K值性能比较**
4. **不同距离度量比较**
5. **交叉验证评估**
6. **结果可视化与分析**

### 3. 自定义参数

```python
# 设置不同的K值进行比较
k_values = [1, 3, 5, 7, 9]

# 选择距离度量方法
distance_metrics = ['euclidean', 'manhattan', 'cosine']

# 设置训练集比例
train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
```

## 实验结果

### 主要发现

1. **K值影响**: 不同数据集的最优K值不同，需要通过交叉验证确定
2. **距离度量**: 欧几里得距离在大多数情况下表现最佳
3. **训练集大小**: 训练集比例对性能有显著影响
4. **数据集特性**: 高维数据(USPS)和低维数据(Iris)的表现差异

### 性能指标

- Iris数据集: 最高准确率可达95%+
- Sonar数据集: 最高准确率约85%
- USPS数据集: 最高准确率约95%

## 可视化功能

- **准确率曲线**: OA、AA、Kappa随K值变化
- **混淆矩阵热力图**: 详细的分类结果展示
- **训练集比例分析**: 不同训练比例的性能对比
- **交叉验证结果**: 多折验证的稳定性分析

## 作业要求完成情况

✅ **算法实现**: K近邻分类和最近邻分类  
✅ **多数据集验证**: USPS、Sonar、Iris三个数据集  
✅ **参数分析**: 不同K值和距离度量的比较  
✅ **训练集比例分析**: 绘制错误率/正确率曲线  
✅ **完整报告**: 详细的实验过程和结果分析  
✅ **代码实现**: 完整的Jupyter Notebook实现  

## 作者信息

- **课程**: 模式识别
- **学校**: 西安电子科技大学
- **作业**: 第一次大作业 - KNN算法实现与分析

# 
