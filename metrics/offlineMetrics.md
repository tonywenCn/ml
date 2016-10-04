如果进行数据建模或者优化工作，在最开始需要思考的问题之一就是如何定义指标，指标就是对改进的量化,来衡量是否有改进以及改进的大小，一般情况根据是否是使用真实流量可以分成下面2类：
- **离线指标**：不通过真实用户真实反馈，而是通过离线的评估，如人工打标的相关性或者利用历史的点击/购买数据来评估想法的有效性。
- **在线指标**：通过真实的用户反馈获取到的数据，在互联网公司中通常采用ABTest的方法。

在工作中指标定义是非常重要的，只有指标定义清晰，才能比较容易的反应出我们的方法是否有效，因此指标设计是数据工作者的核心工作之一。 本文主要聚焦在离线指标的设计，总结了常见的离线指标。



# 分类指标
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

## 精度(Accuracy) 和 错误 (Error)
这个指标既适合二分类问题，也适合都分类问题。

**精度定义**如下：

![](http://latex.codecogs.com/gif.latex?Accuracy=\\frac{1}{N} \\sum_{n=1}^N(f(x_i)=y_i)=\\frac{TP + TN}{TP + TN + FP + FN})

其表达的是: 预测正确的占整体样本的比例。在上面相关性的例子中，我们可以计算得到精度为：![](http://latex.codecogs.com/gif.latex?Accuracy=\\frac{TP + TN}{TP + TN + FP + FN}=\\frac{600 + 250}{600 + 100 + 50 + 250} = 0.85)


**错误定义**如下：

![](http://latex.codecogs.com/gif.latex?Error=\\frac{1}{N} \\sum_{n=1}^N(f(x_i)\\neq y_i)=\\frac{FP + FN}{TP + TN + FP + FN}=1-Accuracy)

其表达的是：预测错误的占整体样本的比例。在上面相关性的例子中，我们可以计算得到错误率为：![](http://latex.codecogs.com/gif.latex?Error=\\frac{TP + TN}{TP + TN + FP + FN}=\\frac{100 + 50}{600 + 100 + 50 + 250} = 0.15)

---

## 准确率(Precision)、 召回 (Recall) 和 F1
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

F1 score是precision和recall的调和平均(较为容易证明F1的取值范围为[min(precision, recall), max(precision, recall)])。 从定义中也较为容易的看出：F1 score是既考虑了precision也考虑了recall，这样我们在比较2个算法的好坏的时候，只需要一个单一的标量即可。

**PR曲线**

绝大多数分类器，能预测一个连续的值，预测结果在不同threshold下，可以获得不同的precision和recall值， 于是我们可以画出precision recall曲线，然后我们可以根据业务需求和precision recall曲线，选择具体的threshold来判定正负例。下面是一个典型的precision recall曲线

![](https://qph.ec.quoracdn.net/main-qimg-ddd56eeeae45bcd95093859b87454e73?convert_to_webp=true)


---

## ROC曲线和面积
**Precision/Recall/F1的缺陷**

在precision、recall中，没有考虑TN的情况(precision, recall & F1只考虑了 TP FP FN的情况)， 在class distribution imbalance problem中，其不是很好的分类指标，例如，分类1有90个samples,分类2有10个samples, 如果:
- 分类器1：把所有samples都预测为正例，其precision=0.9, recall=1.0, F1=0.947
- 分类器2：把分类1 90个samples中的70个分为正例, 分类2 10个samples中的5个分为了正例， 其precision=70/(70+5)=0.93, recall=70/90=0.78, F1=0.848

分类器1虽然F1 score高于分类器2， 但是从实际中分类器2更有用，因为分类器1对类别2完全没有分类能力。(**TODO:如何定义更有用**)。 

**ROC曲线定义**
> In statistics, a receiver operating characteristic (ROC), or ROC curve, is a graphical plot that illustrates the performance of a binary classifier system as its discrimination threshold is varied.  The curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings.

ROC曲线：是一条曲线，其中:
- 横坐标是FPR(False Positive Rate): 

   ![](http://latex.codecogs.com/gif.latex?FPR=\\frac{FP}{FP + TN})      
   FPR可以理解为是负例的召回率
   
- 纵坐标是TPR(True Positive Rate):  

   ![](http://latex.codecogs.com/gif.latex?FPR=\\frac{TP}{TP + FN})       
   TPR可以理解为是正例的召回率
   
在ROC曲线中，每个点(x,y)表示的含义是: 在正例召回率为y的情况下， 负例召回率为x。因为x/y表达的都是召回率，其取值范围为[0,1], 因此ROC曲线是一个在1x1的正方形中的曲线。 对二分的问题进行分类：
- 无任何分类能力的分类器：如果对所有samples赋一个随机数，对于FPR=x, TPR的期望也是x, 这时ROC曲线是一条y=x的直线。 
- 一个完美的分类器是：在正例召回率(TPR)为1.0时, 负例的召回率(FPR)为0.0, 这个时候ROC曲线是2条直线组成(a) 从(0,0)到(0,1)的直线 (b) 从(0,1)到(1,1)的直线。

ROC曲线上的点比较直观的刻画了分类器在某个阈值下，正负例各自的召回比例，反映出在这个阈值下分类器的分类能力。 但是在评估不同的模型时，如果使用ROC来评估哪个模型更好，最好要比较直观的通过1个标量数据来表达出来，在实践中，这个指标就是ROC曲线下的面积(Area Under Curve: AUC)。

![ROC曲线](https://upload.wikimedia.org/wikipedia/commons/6/6b/Roccurves.png)

aucROC是有其物理意义的，在wikipedia上的解释如下: aucROC是等价于随机从正负例中各抽样一个sample， 正例的得分高于负例的得分的概率就等价于aucROC的值，具体请参考wikipedia:
> When using normalized units, the area under the curve (often referred to as simply the AUC, or AUROC) is equal to the probability that a classifier will rank a randomly chosen positive instance higher than a randomly chosen negative one (assuming 'positive' ranks higher than 'negative').
 
ROC曲线有以下特点：
- 随着FPR的增加，TPR是增加或者持平的，TPR不可能降低； 也就是ROC是一条随x轴增大，y递增或者持平的曲线。
- 如果aucROC低于0.5, 只需要将正负样本的label改变以下即可得到比随机更好的分类器。

## PR & ROC曲线的关系
PR和ROC的曲线的关系是什么？ 从同一个confusion matrix得到的2个指标，如何来比较呢？ [2006年ICML一篇文章试图来回答这个问题](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2006_DavisG06.pdf), 总结如下：
- 给定同一个测试集合(正负样本数量是固定的), PR上任何一个点都对应ROC上一个确定的点， 反之亦然，也就是给定PR曲线，ROC曲线也就固定了。
- 针对同一个测试结合：
   - 模型1的ROC曲线每个点都高于模型2，在PR曲线中，模型1的每个点都高于模型2。
   - 模型1的PR曲线每个点都高于模型2，在ROC曲线中，模型1的每个点都高于模型2。
- ROC满足convex hull特征，因此可以使用线性插值来计算AUC； PR 则不能使用线性插值来计算AUC
- 在正负样本差异很大时，aucPR比ROC更加可靠(**TODO: 再次确认**)

# 排序指标
排序指标一般用来衡量序关系的好坏，常见的排序指标包含:
- DCG(Discounted Cumulative Gain) 和 NDCG(Normalized DCG)
- MRR(Mean Reciprocal Rank)
- MAP(Mean Average Precision)
- MAPE(Mean Absolute Percentage Error)
- Kendall Tau's

## DCG & NDCG
在搜索相关性排序中，DCG和NDCG是最经常使用的排序指标，主流的搜索引擎公司最核心的指标之一就是DCG&NDCG。 
在排序中，一般会有如下假设：
- 更高相关性的文档排名越靠前，排序效果越好。
- 在同一个位置上，高相关性的文档比不相关商品排名要好。

于是根据这个假设，就设计出DCG：

![](http://latex.codecogs.com/gif.latex?DCG@p=rel_1+\\sum_{i=2}^p(\\frac{rel_i}{log_2(i)})  (1))

![](http://latex.codecogs.com/gif.latex?DCG@p=\\sum_{i=1}^p(\\frac{2^{rel_i} - 1}{log_2(i+1)})  (2))

其中(1)跟(2) 比较，差异在于:
1. 相同位置和质量的文档，(2)对position的discount更大。
2. 相同位置和质量的文档，(2)对质量的文档加权更大。

在工业界一般采用(2), 因为(2)趋向给把更相关商品排在更前面的算法更高分数。

在使用DCG计算排序结果的指标时，面临一个问题就是不同Query之间的DCG是不可以比的，因为不同Query的相关和不相关的文档数量是不一样的。于是就引入了NDCG(normalized DCG):

![](http://latex.codecogs.com/gif.latex?nDCG@p=sum_{i=1}^N\\frac{DCG_i}{IDCG_i}))

其中![](http://latex.codecogs.com/gif.latex?IDCG_i)是最好的排序结果时的DCG，即按照相关性由高到低排序后计算得到的DCG。这样NDCG就是一个(0,1]之间的一个指标，值越大，排序效果越好。


## MRR （Mean Reciprocal Rank）
MRR经常使用在二分类问题的排序指标上。 定义请参考[wikipedia](https://en.wikipedia.org/wiki/Mean_reciprocal_rank): 
> The mean reciprocal rank is a statistic measure for evaluating any process that produces a list of possible responses to a sample of queries, ordered by probability of correctness. The reciprocal rank of a query response is the multiplicative inverse of the rank of the first correct answer. The mean reciprocal rank is the average of the reciprocal ranks of results for a sample of queries Q.

![](http://latex.codecogs.com/gif.latex?MRR=\\frac{1}{|Q|}\\sum_{i=1}^{|Q|}(\\frac{1}{Rank_i}))

其中![](http://latex.codecogs.com/gif.latex?Rank_i) 表示第i个Query中第一个相关的文档的位置.

## MAP (Mean Average Precision)
MAP经常使用在二分类问题的排序指标上。 
- P@k: 表示1个Query下，top k个document的precision.

   ![](http://latex.codecogs.com/gif.latex?P@k=\\frac{\\sum_{i=1}^k(rel_i)}{k})
- AvgP@k：表示1个Query下，不同位置生成的PR曲线的面积(AUC):

   ![](http://latex.codecogs.com/gif.latex?AvgP@k=\\frac{\\sum_{i=1}^k(P@i * rel(i))}{num\\ of\\ relevant\\ document\\ in\\ top\\ k})
- MAP@k: 表示N个Query的平均AvgP@k

   ![](http://latex.codecogs.com/gif.latex?MAP@k=\\frac{\\sum_{i=1}^N(AvgP@k)}{N})
   
因此MAP的物理意义就是：每个Query平均的aucPR.

## Kendall Tau's
Kendall's tau在实际中较少使用(在Y!的项目中使用过该指标)，主要用于描述2个随机变量的相关性。这里可以用来表示1个perfect的序关系和一个预测的序关系的相关性。 具体可以参考[Kendall Tau's ranking correlation](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient)

# 回归指标
## MAE（Mean Absolute Error）
![](http://latex.codecogs.com/gif.latex?MAE=\\frac{\\sum_{i=1}^N(|y_i - \\hat{y_i}|)}{N})

## MedAE （Median Absolute Error）
MAE中有一个比较大的问题是，某些outlier会贡献绝对部分的Absolute Error, 为了克服这个问题，引入了MedAE:
![](http://latex.codecogs.com/gif.latex?MedAE=median(|y_1 - \\hat{y_1}|, |y_2 - \\hat{y_2}|... |y_N - \\hat{y_N}|))

## MSE（Mean Square Error）
![](http://latex.codecogs.com/gif.latex?MSE=\\frac{\\sum_{i=1}^N(y_i - \\hat{y_i})^2}{N})

## RMSE （Root Mean Square Error）
![](http://latex.codecogs.com/gif.latex?RMSE=\\sqrt[2]{\\frac{\\sum_{i=1}^N(y_i - \\hat{y_i})^2}{N}})

## MAPE （Mean Absolute Percentag Error）
![](http://latex.codecogs.com/gif.latex?MAPE=100 * \\frac{1}{N} \\sum_{i=1}^N\\frac{|y_i - \\hat{y_i}|}{|y_i|})


# Reference:
1. [DCG & NDCG wikipedia](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)
2. [MAPE wikipedia](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)
3. [MRR wikipedia](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)
4. [F1 score wikipedia](https://en.wikipedia.org/wiki/F1_score)
5. [ICML2006: The Relationship Between Precision-Recall and ROC Curves](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2006_DavisG06.pdf)
6. [参考的一篇中文博客，介绍ROC:AUC(Area Under roc Curve )计算及其与ROC的关系](http://www.cnblogs.com/guolei/archive/2013/05/23/3095747.html)
7. [ROC wikipedia](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
8. [常见Ranking指标](http://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf)

