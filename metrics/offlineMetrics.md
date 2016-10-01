如果我们进行数据建模或者优化工作，在最开始需要思考的问题之一就是如何定义指标，指标就是对改进的量化，一般情况根据是否是使用真实流量可以分成下面2类：
- **离线指标**：不通过真实用户真实反馈，而是通过离线的评估，如人工打标的相关性或者利用历史的点击/购买数据来评估想法的有效性。
- **在线指标**：通过真实的用户反馈获取到的数据，在互联网公司中通常采用ABTest的方法。

在工作中指标定义是非常重要的，只有指标定义清晰，才能比较容易的反应出我们的方法是否有效，因此指标设计是数据工作者的核心工作之一。 本文主要聚焦在离线指标的设计，总结了常见的离线指标。

# 回归指标

# 分类指标
## 二分类
二分类问题对应的指标基本和分类准确与否相关，而分类准确与否按照真实类别和预测类别这两个维度可以分成4个象限:

| 真实类别        | 预测类别   | 预测类别 |
| ------------- |:-------------:| -----:|
|     | 正例 | 负例 |
|正例 | TP(True Positive)   | FN(False Negative)   |
|负例 | FP(False Positive)   | TN(True Negative)   |

简单解释一下上面这个表格，TP表示_正例被预测成正例_， 其他类推。

根据上面这个表格，可以计算下面这些指标：
- 精度(Accuracy) 和 错误 (Error)
- 准确率(Precision)、 召回 (Recall) 和 F1
- 曲线和面积：aucROC 和 aucPR

举个例子，如果有1000\<Query, Document>对，其中：
- 有700对人工评估是相关的，
- 有300对人工评估是不相关的。
有一个相关性模型能对这些\<Query, Document>对进行相关性评分，评分后的结果如下：

| 真实类别        | 预测类别   | 预测类别 |
| ------------- |:-------------:| -----:|
|     | 正例 | 负例 |
|正例 | TP(600对)   | FN(100对)   |
|负例 | FP(50对)   | TN(250对)   |

下面详细介绍这些指标。

---

### 精度(Accuracy) 和 错误 (Error)
这个指标既适合二分类问题，也适合都分类问题。

**精度定义**如下：

![](http://latex.codecogs.com/gif.latex?Accuracy=\\frac{1}{N} \\sum_{n=1}^N(f(x_i)=y_i)=\\frac{TP + TN}{TP + TN + FP + FN})

其表达的是: 预测正确的占整体样本的比例。在上面相关性的例子中，我们可以计算得到精度为：![](http://latex.codecogs.com/gif.latex?Accuracy=\\frac{TP + TN}{TP + TN + FP + FN}=\\frac{600 + 250}{600 + 100 + 50 + 250} = 0.85)


**错误定义**如下：

![](http://latex.codecogs.com/gif.latex?Error=\\frac{1}{N} \\sum_{n=1}^N(f(x_i)\\neq y_i)=\\frac{FP + FN}{TP + TN + FP + FN}=1-Accuracy)

其表达的是：预测错误的占整体样本的比例。在上面相关性的例子中，我们可以计算得到错误率为：![](http://latex.codecogs.com/gif.latex?Error=\\frac{TP + TN}{TP + TN + FP + FN}=\\frac{100 + 50}{600 + 100 + 50 + 250} = 0.15)

---

### 准确率(Precision)、 召回 (Recall) 和 F1
准确率和召回率是在IR中使用最多的指标。

**准确率定义**：

![](http://latex.codecogs.com/gif.latex?Precision=\\frac{TP}{TP + FP})

**召回率定义**:

![](http://latex.codecogs.com/gif.latex?Precision=\\frac{TP}{TP + FN})

**最常用的场景**： 在给定样本集合上，有一个分类器给出一个相关性的连续预测值(例如LR/GBDT)，可以取了一个阈值，大于等于该阈值的样本被称为预测为相关，小于该阈值的被称为预测不相关，于是我们就可以计算当前阈值下的准确率和召回率。 还使用上面这个相关性的问题作为例子：
- ![](http://latex.codecogs.com/gif.latex?Precision=\\frac{TP}{TP + FP}=\\frac{600}{600+50}=0.923)
- ![](http://latex.codecogs.com/gif.latex?Precision=\\frac{TP}{TP + FN}=\\frac{600}{600+100}=0.857)



**F1定义**

![](http://latex.codecogs.com/gif.latex?F1=\\frac{2}{\\frac{1}{Precision} + \\frac{1}{Recall}})

F1 score是precision和recall的调和平均。 从定义中也较为容易的看出：F1 score是既考虑了precision也考虑了recall，这样我们在比较2个算法的好坏的时候，只需要一个单一的标量即可。


---

### ROC曲线面积和PR面积
**P/R/F1的缺陷**

在precision、recall中，我们没有考虑TN的情况(precision, recall & F1只考虑了 TP FP FN的情况)， 在class distribution imbalance problem中，其不是很好的分类指标，例如，分类1有90个samples,分类2有10个samples, 如果:
- 分类器1：把所有samples都预测为正例，其precision=0.9, recall=1.0, F1=0.947
- 分类器2：把分类1 90个samples中的70个分为正例, 分类2 10个samples中的5个分为了正例， 其precision=70/(70+5)=0.93, recall=70/90=0.78, F1=0.848

分类器1虽然F1 score高于分类器2， 但是从实际中分类器2更有用(**TODO:如何定义更有用**)。 

**ROC曲线定义**


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
4. [F1 score wikipedia](https://en.wikipedia.org/wiki/F1_score)
5. [ICML2006: The Relationship Between Precision-Recall and ROC Curves](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2006_DavisG06.pdf)
6. [参考的一篇中文博客，介绍ROC:AUC(Area Under roc Curve )计算及其与ROC的关系](http://www.cnblogs.com/guolei/archive/2013/05/23/3095747.html)
