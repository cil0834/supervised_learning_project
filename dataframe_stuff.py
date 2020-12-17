import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data = pd.read_csv("forestfires.csv")


train, test = train_test_split(data, test_size=0.2)
train, validation = train_test_split(train, test_size=0.25)
test_y = test['area']
validation_y = validation['area']


train.to_csv('train.csv')
test.to_csv('test.csv')
validation.to_csv('validation.csv')