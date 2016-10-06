import sys
import numpy as np

class linear_regression(object):
    def __init__(self):
        self.max_iter = 100
        self.tol = 1e-5
        self.sample_cnt = -1
        self.feature_dim = -1
        self.coef = np.array(())
        self.grad = np.array(())
        return 

    def gradient(self, x, y, w):
        return np.dot(x.transpose(), np.dot(x, w) - y)

    def loss(self, x, y, w):
        return np.sum(np.power(np.dot(x, w) - y, 2))
    
    def predict(self, x):
        return np.dot(x, self.coef)

    def fit(self, x, y, max_iter = 100, tol = 1e-5):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return -1
        
        if len(x.shape) != 2 or len(y.shape) != 1 or x.shape[0] != y.shape[0]:
            return -1

        self.max_iter = max_iter
        self.tol = tol
        self.sample_cnt = y.shape[0]
        self.feature_dim = x.shape[1]
        self.coef = np.random.uniform(-1, 1, self.feature_dim)
        self.grad = self.gradient(x, y, self.coef)
        loss = self.loss(x, y, self.coef)

        step_size = 1.0
        for i in xrange(self.max_iter):
            grad_new = self.gradient(x, y, self.coef)
            w_new = self.coef - step_size * grad_new
            loss_new = self.loss(x, y, w_new)
            
            # Line search
            if loss < loss_new:
                step_size = step_size * 0.5
            else:
                step_size = step_size * 2
            
            self.grad = grad_new
            loss = loss_new
            self.coef = w_new
            print self.coef
            print "loss:", loss
            
        return 0

