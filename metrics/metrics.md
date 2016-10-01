# 回归指标

# 分类指标
## 二分类
二分类问题的指标基本都围绕下面这个表格展开

| 真实类别        | 预测类别   | 预测类别 |
| ------------- |:-------------:| -----:|
|     | 正例 | 负例 |
|正例 | TP   | FN   |
|负例 | FP   | TN   |

- 精度(Accuracy) 和 错误 (Error)
- 准确率(Precision)、 召回 (Recall) 和 F1
- aucROC 和 aucPR

## 多分类指标

# 排序指标
排序指标一般用来衡量序关系的好坏，常见的排序指标包含:
- DCG(Discounted Cumulative Gain) 和 NDCG(Normalized Discounted Cumulative Gain)
- MRR(Mean Reciprocal Rank)
- MAP(Mean Average Precision)
- MAPE(Mean Absolute Percentage Error)

# 聚类指标


# Reference:
1. [DCG & NDCG wikipedia][https://en.wikipedia.org/wiki/Discounted_cumulative_gain]
2. [MAPE wikipedia][https://en.wikipedia.org/wiki/Mean_absolute_percentage_error]
3. [MRR wikipedia][https://en.wikipedia.org/wiki/Mean_reciprocal_rank]
