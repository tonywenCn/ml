
# 简介

## Continous Bag of Words（CBOW）

连续词袋模型(Continous Bag Of Words)的问题定义：

* 在一篇文章中(文章按照term长度为m)，使用一个滑动窗口(滑动窗口大小为2n + 1)从前到后滑动，生成m - (2n + 1)个片段。
* CBOW模型预测：对1个连续片段，给定前n个term 和  后n个term, 预测中间的term。
* 建模：
   * 输入：V纬度(V是term字典大小)，其输入向量I中的每个元素I[k]表示单词k在当前连续片段中出现还是没有没有出现，出现用1表示，未出现用0表示。
   * **隐藏层**：采用1个隐藏层的全连接神经网络，其隐藏层的大小一般从几十到千。
   * **输出层**：V纬度(V是term字典大小)，其输出向量O表示给定一个片段中的前n和后n个term, 其中间的term的概率。

  
## Skip Gram

1. Item
2. Item
   * Mixed
   * Mixed  
3. Item

# 

# Reference:
1. [word2vec Parameter Learning Explained](http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf)
