import numpy as np
from matplotlib import pyplot as plt
from linear_regression import linear_regression

from sklearn import linear_model, datasets


n_samples = 1000
n_outliers = 50


X = np.random.uniform(-10,10,1000)
y = X
X = X + X *  np.random.normal(0, 0.2, 1000)
X = X.reshape((1000, 1))

print X.shape, y.shape
model = linear_regression()
model.fit(X, y)
pred = model.predict(X)

print model.coef
plt.scatter(X, y, color='gold', marker='.')
plt.plot(X.reshape((1000)), pred)
plt.grid(True)
plt.show()

