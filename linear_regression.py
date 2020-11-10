import pandas as pd
import numpy as np

a = pd.read_csv("forestfires.csv")

del a['X']
del a['Y']
del a['month']
del a['day']

Y = a['area']
del a['area']



# Multiple linear regression
b_0 = np.ones((517, 1))
X = a.to_numpy()
X = np.c_[b_0, X]
X_t = X.transpose()
X_t_X = np.matmul(X_t, X)
X_t_Y = np.matmul(X_t,Y)
X_t_X_i = np.linalg.inv(X_t_X)
B = np.matmul(X_t_X_i, X_t_Y)

# Ridge Regression


class Regression:
    def __init__(self, data, y, x, regression_type= "linear"):
        self.type = regression_type
        self.data = data
        self.y = y
        self.x = x



