如果我们进行数据建模或者优化工作，在最开始需要思考的一个问题就是如何定义指标，指标就是对工作的数字量化，指标根据是否是真实流量可以分成2个：
- 离线指标：不通过真实用户真实反馈，而是通过离线的评估，如人工打标的相关性或者利用历史的点击/购买数据来评估想法的有效性。
- 在线指标：通过真实的用户反馈获取到的数据，在互联网公司中通常采用ABTest的方法。

在工作中指标定义是非常重要的，只有指标定义清晰，才能比较容易的反应中我们的方法是否有效，因此指标是数据工作者的核心。 本文主要聚焦在离线指标的设计，汇总了常见的离线指标。

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

**精度定义**如下：

![](http://latex.codecogs.com/gif.latex?Accuracy=\\frac{1}{N} \\sum_{n=1}^N(f(x_i)=y_i)=\\frac{TP + TN}{TP + TN + FP + FN})

其表达的是: 预测正确的占整体样本的比例。


**错误定义**如下：

![](http://latex.codecogs.com/gif.latex?Error=\\frac{1}{N} \\sum_{n=1}^N(f(x_i)\\neq y_i)=\\frac{FP + FN}{TP + TN + FP + FN}=1-Accuracy)

其表达的是：预测错误的占整体样本的比例。
### 准确率(Precision)、 召回 (Recall) 和 F1
准确率和召回率是在IR中使用最多的指标。

**准确率定义**：

![](http://latex.codecogs.com/gif.latex?Precision=\\frac{TP}{TP + FP})

**召回率定义**:

![](http://latex.codecogs.com/gif.latex?Precision=\\frac{TP}{TP + FN})

**F1定义**

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
