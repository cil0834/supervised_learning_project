import pandas as pd
import numpy as np
from Decision_Tree import Regression_Tree
import random
from sklearn.model_selection import train_test_split
import math

class Random_Forest:
    def __init__(self, n_attributes, max_depth, min_values, data, x, forest_size, response):
        """

        :param n_attributes: the number of attributes taken from the total list of attributes
        :param max_depth: The max depth of the tree
        :param min_values: The minimum values allowed in a single node
        :param data: The data that is being used
        :param x: The independent variables
        :param response: The response
        """
        self.n_attributes = n_attributes
        self.forest_size = forest_size
        self.max_depth = max_depth
        self.min_values = min_values
        self.X = data[x].copy()
        self.x_columns = self.X.columns
        self.forest = []
        self.response = response
        self.Y = data[response]
        self.data = data

    def build_tree(self):
        """
        The function that builds a tree to be added to the forest
        :return:
        """
        indexes = list(range(len(self.x_columns)))
        chosen_indexes = random.sample(indexes, self.n_attributes)
        attributes = self.x_columns[chosen_indexes]
        tree = Regression_Tree(max_depth=self.max_depth, min_values=self.min_values, data=self.data, x = attributes, response=self.response)
        tree.train_tree()
        self.forest.append(tree)

    def build_forest(self):
        """
        Builds a forest based on the number of trees you want
        :return:
        """
        for i in range(self.forest_size):
            self.build_tree()

    def predict_individual(self, data_point, regression_tree):
        """
        A function that makes a prediction for a tree at a specific data point
        :param data_point: The data point being predicted
        :return: The predicted value for the data point
        """
        prediction = None
        branch = regression_tree.tree
        while prediction == None:
            category = branch['category']
            threshold = branch['threshold']

            if data_point[category] < threshold:
                branch = branch['left_branch']
            else:
                branch = branch['right_branch']
            prediction = branch.get("predicted_value")
        return(prediction)

    def predict(self, x_values):
        """
        The values that are going to be predicted
        :param x_values: A data frame of predicted values
        :return: An array of predicted values
        """
        predictions = []
        for x_value in x_values.iloc:
            current_predictions = 0
            for tree in self.forest:
                prediction = self.predict_individual(x_value, tree)
                current_predictions += prediction
            current_predictions = current_predictions / self.forest_size
            predictions.append(current_predictions)
        return predictions

    def calculate_error(self, predictions, targets):
        """
        Calculates the residual sum of squares
        :param prediction: the predicted values
        :param target: the target values
        :return: Residual sum of squares
        """
        rss = 0
        for prediction, target in zip(predictions, targets):
            rss += ((prediction - target) ** 2)
            #print("Prediction: ", prediction, " Target: ", target)
        return math.sqrt(rss/len(predictions))

data = pd.read_csv("forestfires.csv")
Xs= ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
data['area'] = np.log2(data['area'] + 1)
train, test = train_test_split(data, test_size=0.2)
train, validation = train_test_split(train, test_size=0.25)
test_y = test['area']



abc = Random_Forest(4, 5, 3, data, Xs, 100, 'area')

#abc.build_forest()
#predictions = abc.predict(x_vals)
#error = abc.calculate_error(predictions, y_vals)
#print(error)


best_error = np.inf
best_size = 0

f_5 = Random_Forest(4, 3, 10, data, Xs, 5, 'area')
f_5.build_forest()
prediction = f_5.predict(validation)
error = f_5.calculate_error(prediction, validation['area'])
print(" The forest size is: 5", " The error is: ", error)

f_10 = Random_Forest(4, 3, 10, data, Xs, 10, 'area')
f_10.build_forest()
prediction = f_10.predict(validation)
error = f_10.calculate_error(prediction, validation['area'])
print(" The forest size is: 10", " The error is: ", error)

f_50 = Random_Forest(4, 3, 10, data, Xs, 50, 'area')
f_50.build_forest()
prediction = f_50.predict(validation)
error = f_50.calculate_error(prediction, validation['area'])
print(" The forest size is: 50", " The error is: ", error)

f_100 = Random_Forest(4, 3, 10, data, Xs, 100, 'area')
f_100.build_forest()
prediction = f_100.predict(validation)
error = f_100.calculate_error(prediction, validation['area'])
print(" The forest size is: 100", " The error is: ", error)

f_200 = Random_Forest(4, 3, 10, data, Xs, 200, 'area')
f_200.build_forest()
prediction = f_200.predict(validation)
error = f_200.calculate_error(prediction, validation['area'])
print(" The forest size is: 200", " The error is: ", error)