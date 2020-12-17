import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import r2_score


def calculate_error(predictions, targets):
    """
    Calculates the residual sum of squares
    :param prediction: the predicted values
    :param target: the target values
    :return: Residual sum of squares
    """
    rss = 0
    for prediction, target in zip(predictions, targets):
        rss += ((prediction - target) ** 2)
    return math.sqrt(rss / len(predictions))



Xs= ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
train = pd.read_csv('train.csv')
validation = pd.read_csv('validation.csv')
test = pd.read_csv('test.csv')

train['area'] = np.log2(train['area'] + 1)
validation['area'] = np.log2(validation['area'] + 1)
test['area'] = np.log2(test['area'] + 1)
training = [train, validation]
training = pd.concat(training)

linear_model = LinearRegression()
linear_model.fit(training[Xs], training['area'])

predictions = linear_model.predict(test[Xs])

error = calculate_error(predictions, test['area'])
print(error)