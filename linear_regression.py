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



data = pd.read_csv("forestfires.csv")

Xs= ['temp', 'RH', 'wind', 'rain']
data['area'] = np.log2(data['area'] + 1)

train, test = train_test_split(data, test_size=0.2)
linear_model = LinearRegression()
linear_model.fit(train[Xs], train['area'])

predictions = linear_model.predict(test[Xs])

error = calculate_error(predictions, test['area'])
print(error)

r_squared = r2_score(test['area'], predictions)

print(r_squared)