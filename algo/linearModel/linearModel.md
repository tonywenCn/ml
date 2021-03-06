线性模型是日常工作中使用最多的模型，原因是线性模型足够简单：
- 模型训练简单&高效
- 模型参数和理解简单

因为上面这2个原因，使得线性模型非常的流行，尤其在工业界，如广告系统中大量采用logistic regression。

# Linear Regression
- **输入**： ![](http://latex.codecogs.com/gif.latex?(x_1,y_1), (x_2, y_2), (x_3, y_3)...(x_n, y_n)) ， 其中xi是特征向量，yi是观测值（样本label）
- **输出**：w向量 使得 
   
   ![](http://latex.codecogs.com/gif.latex?w=min(f(x))) 
   
   ![](http://latex.codecogs.com/gif.latex?f(w,x)=\\sum_{i=1}^n(\\sum_{j=1}^m(w_j*x_{ij}) - y_i)^2) 

linear regression的直观解释:
- loss function表达的是：样本点到直线wx的距离。
- 最小化求解得到的直线wx是使所有样本点到直线wx欧几里得距离最小的直线。

这是一个典型无约束优化问题，该求解函数满足：
- 连续的，处处可导的函数
- convex 函数(**TODO:证明这是convex函数**)

这个优化问题最简单的数值求解方法就是使用梯度下降：
- ![](http://latex.codecogs.com/gif.latex?w_j=w_j - \\gamma \\frac{\\partial f}{\\partial w_j} ) 
- 其中：
   - ![](http://latex.codecogs.com/gif.latex?\\gamma)是每次迭代的步长
   - ![](http://latex.codecogs.com/gif.latex?\\frac{\\partial f}{\\partial w_j}) 是loss function关于w的偏导数:
      
      ![](http://latex.codecogs.com/gif.latex?\\frac{\\partial f}{\\partial w_j}=\\sum_{i=1}^n(x_{ij}(\\sum_{j=1}^m(w_j*x_{ij}) - y_i)))

**例子**
- git source code: https://github.com/tonywenCn/ml/blob/master/algo/linearModel/src/linear_regression.py
- exmaple: ![](src/img/linear_regression_test1.png)

# Logistic Regression
对于二分类问题的拟合最容易想到方法是：
- 把分类标签作为target 0/1， 然后使用linear regression进行拟合。在预测中，如果预测值大于0.5, 则标记为类别1，否则标记为类别0.

该方法是比较直观的，


# LDA

# 总结

# Reference
