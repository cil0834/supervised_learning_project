import pandas as pd
import numpy as np
import math

a = pd.read_csv("forestfires.csv")

del a['X']
del a['Y']
del a['month']
del a['day']

#print(a)
'''
Y = a['area']
Y = Y + 1
Y = np.log(Y)
'''
#print(Y)

#del a['area']
'''
# Multiple linear regression
b_0 = np.ones((517, 1))
X = a.to_numpy()
X = np.c_[b_0, X]
X_t = X.transpose()
X_t_X = np.matmul(X_t, X)
X_t_Y = np.matmul(X_t,Y)
X_t_X_i = np.linalg.inv(X_t_X)
B = np.matmul(X_t_X_i, X_t_Y)
'''
# Ridge Regression



class Linear_Regression:
    def __init__(self, data, x, y):
        """
        :param data: dataframe of the data with response and independent variables
        :param y: The name of the response
        :param x: A list of names of the independent variables
        """
        self.data = data
        Y = self.data[y]
        Y = Y + 1
        self.Y = np.log(Y)

        index = self.data.index
        number_rows = len(index)
        data = self.data[x].copy()
        X = data.to_numpy()
        b_0 = np.ones((number_rows, 1))
        X = np.c_[b_0, X]

        self.X = X
        self.n = number_rows
        self.k = len(x)

    def cal_weights(self):
        """
        This function calculates the weights
        :return: A weight vector
        """
        X_t = self.X.transpose()
        X_t_X = np.matmul(X_t, self.X)
        X_t_Y = np.matmul(X_t, self.Y)
        X_t_X_i = np.linalg.inv(X_t_X)
        weights = np.matmul(X_t_X_i, X_t_Y)

        return(weights)

    def Y_hat(self):
        """
        This function calculates Y_hat
        :return: Y_hat prediciton
        """
        weights = self.cal_weights()
        Y_hat = np.matmul(self.X, weights)
        return (Y_hat)

    def SSE(self):
        """
        This function calculates the residual sum of squares
        :return: Residual sum of squares
        """
        Y_hat = self.Y_hat()
        Y_i_Y =  self.Y - Y_hat
        Y_i_Y_t = Y_i_Y.transpose()
        SSE = np.matmul(Y_i_Y_t, Y_i_Y)

        return(SSE)

    def TSS(self):
        """
        This calculates and returns the total sum of squares
        :return: The total sum of squares
        """
        Y_mean = self.Y.mean()
        Y_i_Y = self.Y - Y_mean
        Y_i_Y_t = Y_i_Y.transpose()

        TSS = np.matmul(Y_i_Y_t, Y_i_Y)

        return(TSS)

    def R_squared(self):
        """
        Caluculate the R_squared vavlue (1-SSE/TSS)
        :return: R^2
        """
        SSE = self.SSE()
        TSS = self.TSS()

        R_s = 1 - (SSE/TSS)
        return(R_s)

    def R_squared_adj(self):
        """
        Calculate adjusted R^2
        :return: Return adjusted R^2
        """
        R_s = self.R_squared()
        num = (1 - R_s)*(self.n - 1)
        denom = (self.n - self.k - 1)
        r_adj = 1-(num/denom)

        return(r_adj)

    def get_weigts(self):
        return self.cal_weights()

Xs= ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']

class Ridge_Regression:
    def __init__(self, data, x, y, lam):
        """
        Initialize Ridge Regression method
        :param data: dataframe of the data with response and independent variables
        :param y: The name of the response
        :param x: A list of names of the independent variables
        """
        self.data = data
        Y = self.data[y]
        Y = Y + 1
        self.Y = np.log(Y)
        self.lam = lam

        index = self.data.index
        number_rows = len(index)
        data = self.data[x].copy()
        X = data.to_numpy()
        b_0 = np.ones((number_rows, 1))
        X = np.c_[b_0, X]

        self.X = X
        self.n = number_rows
        self.k = len(x)

    def cal_weights(self):
        """
        This function calculates the weights
        :return: A weight vector
        """
        X_t = self.X.transpose()
        X_t_X = np.matmul(X_t, self.X)
        identity = self.lam*np.identity(len(X_t_X),)
        X_t_X = (X_t_X - identity)
        X_t_Y = np.matmul(X_t, self.Y)
        X_t_X_i = np.linalg.inv(X_t_X)
        weights = np.matmul(X_t_X_i, X_t_Y)

        return (weights)

    def Y_hat(self):
        """
        This function calculates Y_hat
        :return: Y_hat prediciton
        """
        weights = self.cal_weights()
        Y_hat = np.matmul(self.X, weights)
        return (Y_hat)

    def SSE(self):
        """
        This function calculates the residual sum of squares
        :return: Residual sum of squares
        """
        Y_hat = self.Y_hat()
        Y_i_Y =  self.Y - Y_hat
        Y_i_Y_t = Y_i_Y.transpose()
        SSE = np.matmul(Y_i_Y_t, Y_i_Y)

        return(SSE)

    def TSS(self):
        """
        This calculates and returns the total sum of squares
        :return: The total sum of squares
        """
        Y_mean = self.Y.mean()
        Y_i_Y = self.Y - Y_mean
        Y_i_Y_t = Y_i_Y.transpose()

        TSS = np.matmul(Y_i_Y_t, Y_i_Y)

        return(TSS)

    def R_squared(self):
        """
        Caluculate the R_squared vavlue (1-SSE/TSS)
        :return: R^2
        """
        SSE = self.SSE()
        TSS = self.TSS()

        R_s = 1 - (SSE/TSS)
        return(R_s)

    def R_squared_adj(self):
        """
        Calculate adjusted R^2
        :return: Return adjusted R^2
        """
        R_s = self.R_squared()
        num = (1 - R_s)*(self.n - 1)
        denom = (self.n - self.k - 1)
        r_adj = 1-(num/denom)

        return(r_adj)

#abc = Ridge_Regression(a, Xs, 'area', 0.1)
abd = Linear_Regression(a, Xs, 'area')

print(abd.get_weigts())

print(abd.R_squared())