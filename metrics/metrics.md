# 回归指标

# 分类指标
## 二分类
二分类问题对应的指标基本和分类准确与否相关，而分类准确与否按照真实类别和预测类别这两个维度可以分成4个象限:

| 真实类别        | 预测类别   | 预测类别 |
| ------------- |:-------------:| -----:|
|     | 正例 | 负例 |
|正例 | TP   | FN   |
|负例 | FP   | TN   |

根据上面这个表格，可以计算下面这些指标：
- 精度(Accuracy) 和 错误 (Error)
- 准确率(Precision)、 召回 (Recall) 和 F1
- 曲线和面积：aucROC 和 aucPR

下面详细介绍这些指标。
### 精度(Accuracy) 和 错误 (Error)
这个指标既适合二分类问题，也适合都分类问题。
精度定义如下：

![](http://latex.codecogs.com/gif.latex?Accuracy=\\frac{1}{N} \\sum_{n=1}^N(f(x_i)=y_i))

### 准确率(Precision)、 召回 (Recall) 和 F1
### 曲线和面积：aucROC 和 aucPR

## 多分类指标

# 排序指标
排序指标一般用来衡量序关系的好坏，常见的排序指标包含:
- DCG(Discounted Cumulative Gain) 和 NDCG(Normalized DCG)
- MRR(Mean Reciprocal Rank)
- MAP(Mean Average Precision)
- MAPE(Mean Absolute Percentage Error)

# 聚类指标


# Reference:
1. [DCG & NDCG wikipedia](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)
2. [MAPE wikipedia](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)
3. [MRR wikipedia](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)
